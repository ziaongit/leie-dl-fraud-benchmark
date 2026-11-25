"""Paper 3 v2 - Do sequence / attention deep learning models help detect CONFIRMED Medicare provider fraud?
Benchmark on LEIE-labelled CMS data (Physician & Other Practitioners 2013-2024; Part D Prescribers 2013-2023).

Design choices that answer the IJDSA reviewers:
 - labels: fraud-related LEIE exclusions (1128a1/a3/b7), provider-year 1-3 years before exclusion; negatives never in LEIE
 - splits: provider-grouped K-fold (no provider in train and test), repeated R times; plus temporal split (train 2013-2019, test 2020-2022)
 - every preprocessing step (log transform, standardisation, category vocabularies, clipping) is fitted on the training fold only
 - no SMOTE; class imbalance handled by class weighting in every model; same training subset for every model
 - equal tuning budget: each model gets N_TRIALS random configurations scored on an inner provider-grouped validation split
 - metrics at natural prevalence (negatives weighted by 4): PR-AUC, ROC-AUC, precision/recall at audit budgets k, Brier, ECE
 - CNN-RNN models are run under several random feature orders (order sensitivity); attention vs SHAP/permutation analysed separately
Runs on Kaggle (GPU if available). Env: DATA (input root), WORK (output dir), N_TRIALS, NEG_FRAC, PERMS.
Usage: python p3_bench.py <dataset: phys|partd> <design: grouped|temporal> [reps] [folds]
Results append to results_<dataset>.csv; finished (design, rep, fold, model, perm) rows are skipped on restart."""
import os, sys, time, zlib, glob, json, math
import numpy as np, pandas as pd
from pandas.api.types import union_categoricals
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, xgboost as xgb
import torch, torch.nn as nn

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', DEV, flush=True)
DS, DESIGN = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
FOLDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
MODELS = os.environ.get('MODELS', 'LogReg,LightGBM,XGBoost,MLP,FTT,CNNRNN,AttnCNNRNN').split(',')
N_TRIALS = int(os.environ.get('N_TRIALS', '6'))                # random-search configurations per model (same for every model)
NEG_FRAC = float(os.environ.get('NEG_FRAC', '0.10'))
DATA_FRAC = float(os.environ.get('DATA_FRAC', '1.0'))      # learning curve: share of training PROVIDERS used      # share of sampled training negatives kept (same subset for all models)
PERMS = int(os.environ.get('PERMS', '3'))                  # feature orders for sequence models (first rep only)
WORK = os.environ.get('WORK', '/kaggle/working'); DATA = os.environ.get('DATA', '/kaggle/input')
OUT = f'{WORK}/results_{DS}' + (f'_lc{DATA_FRAC}' if DATA_FRAC < 1 else '') + '.csv'
W_NEG = 4.0                                                # negatives were sampled at 25% of providers

# ---------------- data ----------------
if DS == 'phys':
    files = sorted(glob.glob(f'{DATA}/**/provider_year_*.parquet', recursive=True))
    ID, CAT = 'Rndrng_NPI', ['Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind']
    def load(f):
        t = pd.read_parquet(f); keep = (t['label'] == 1) | (t[ID].map(lambda n: zlib.crc32(n.encode()) % 4 == 0))
        return t[keep].drop(columns=[c for c in t.columns if c.startswith('Bene_Race_')])
else:
    files = sorted(glob.glob(f'{DATA}/**/partd_year_*.parquet', recursive=True))
    ID, CAT = 'NPI', ['cat_Ent_Cd', 'cat_State', 'cat_RUCA', 'cat_Type']
    def load(f): return pd.read_parquet(f)
parts = [load(f) for f in files]
for c in CAT:
    u = union_categoricals([p[c].astype(str).astype('category') for p in parts]).categories
    for p in parts: p[c] = pd.Categorical(p[c].astype(str), categories=u)
d = pd.concat(parts, ignore_index=True); del parts
d = d.drop(columns=[c for c in d.columns if d[c].isna().all()])
y = d['label'].values.astype(np.int8); g = d[ID].values; yr = d['year'].values.astype(int)
X = d.drop(columns=[ID, 'year', 'label', 'years_to_excl']); del d
NUM = [c for c in X.columns if c not in CAT]
XN = X[NUM].values.astype(np.float32); XC = np.stack([X[c].cat.codes.values for c in CAT], 1).astype(np.int32)
NCAT = [len(X[c].cat.categories) for c in CAT]; del X
w_all = np.where(y == 1, 1.0, W_NEG).astype(np.float32)
print(DS, 'rows', len(y), 'pos', int(y.sum()), 'providers', len(set(g)), 'num', len(NUM), 'cat', len(CAT), flush=True)

# ---------------- preprocessing fitted on training rows only ----------------
class Prep:
    def __init__(self, idx, top=50):
        A = np.log1p(np.clip(np.nan_to_num(XN[idx], nan=0.0), 0, None))
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        self.vocab = []                                            # per categorical: training-frequent codes -> index; rest -> 0 (other/unseen)
        for j in range(len(CAT)):
            vc = pd.Series(XC[idx, j]).value_counts(); keep = vc.index[vc >= 20][:top]
            m = np.zeros(NCAT[j] + 1, dtype=np.int32); m[keep.values + 1] = np.arange(1, len(keep) + 1); self.vocab.append(m)
    def num(self, idx):
        A = np.log1p(np.clip(np.nan_to_num(XN[idx], nan=0.0), 0, None)); return np.clip((A - self.mu) / self.sd, -8, 8).astype(np.float32)
    def cat(self, idx): return np.stack([self.vocab[j][XC[idx, j] + 1] for j in range(len(CAT))], 1).astype(np.int64)
    def card(self): return [int(v.max()) + 1 for v in self.vocab]
    def onehot(self, idx):
        C = self.cat(idx); out = []
        for j, k in enumerate(self.card()):
            o = np.zeros((len(idx), k), np.float32); o[np.arange(len(idx)), C[:, j]] = 1; out.append(o)
        return np.hstack([self.num(idx)] + out)

# ---------------- neural models ----------------
class MLP(nn.Module):
    def __init__(s, nn_, card, h=128, drop=0.2, emb=8):
        super().__init__(); s.E = nn.ModuleList([nn.Embedding(k, emb) for k in card])
        s.f = nn.Sequential(nn.Linear(nn_ + emb * len(card), h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(drop), nn.Linear(h // 2, 1))
    def forward(s, xn, xc): return s.f(torch.cat([xn] + [e(xc[:, j]) for j, e in enumerate(s.E)], 1)).squeeze(1)

class Tok(nn.Module):                                         # feature tokenizer shared by FT-Transformer and sequence models
    def __init__(s, nn_, card, dim):
        super().__init__(); s.W = nn.Parameter(torch.randn(nn_, dim) * 0.1); s.b = nn.Parameter(torch.zeros(nn_, dim))
        s.E = nn.ModuleList([nn.Embedding(k, dim) for k in card])
    def forward(s, xn, xc):
        t = xn.unsqueeze(2) * s.W + s.b
        return torch.cat([t] + [e(xc[:, j]).unsqueeze(1) for j, e in enumerate(s.E)], 1)   # (B, F, dim)

class FTT(nn.Module):                                         # order-invariant comparator
    def __init__(s, nn_, card, dim=32, layers=2, heads=4, drop=0.1):
        super().__init__(); s.t = Tok(nn_, card, dim); s.cls = nn.Parameter(torch.zeros(1, 1, dim))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim * 2, drop, batch_first=True), layers)
        s.h = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
    def forward(s, xn, xc):
        t = s.t(xn, xc); t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1); return s.h(s.enc(t)[:, 0]).squeeze(1)

class CNNRNN(nn.Module):                                      # order-dependent: conv over feature axis + GRU (+ optional additive attention)
    def __init__(s, nn_, card, perm, dim=16, ch=32, hid=48, attn=False, drop=0.2):
        super().__init__(); s.t = Tok(nn_, card, dim); s.register_buffer('perm', torch.as_tensor(perm)); s.attn = attn
        s.conv = nn.Sequential(nn.Conv1d(dim, ch, 3, padding=1), nn.ReLU()); s.gru = nn.GRU(ch, hid, batch_first=True)
        if attn: s.a = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1, bias=False))
        s.h = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 1)); s.last_attn = None
    def forward(s, xn, xc):
        t = s.t(xn, xc)[:, s.perm]; h, _ = s.gru(s.conv(t.transpose(1, 2)).transpose(1, 2))
        if s.attn:
            a = torch.softmax(s.a(h).squeeze(2), 1); s.last_attn = a.detach(); z = (a.unsqueeze(2) * h).sum(1)
        else: z = h[:, -1]
        return s.h(z).squeeze(1)

def train_nn(make, P, tri, vai, lr, epochs=10, bs=2048, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xn, Xc, Y = torch.tensor(P.num(tri)), torch.tensor(P.cat(tri)), torch.tensor(y[tri].astype(np.float32))
    pw = (Y == 0).sum() * W_NEG / max(1, (Y == 1).sum())       # balanced class weight in the loss (no resampling)
    m = make().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw), device=DEV))
    Xn, Xc, Y = Xn.to(DEV), Xc.to(DEV), Y.to(DEV)
    best, bstate, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); o = torch.randperm(len(Y), device=DEV)
        for i in range(0, len(Y), bs):
            b = o[i:i + bs]; opt.zero_grad(); l = lossf(m(Xn[b], Xc[b]), Y[b]); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        s = predict_nn(m, P, vai); ap = average_precision_score(y[vai], s, sample_weight=w_all[vai])
        if ap > best: best, bstate, bad = ap, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(bstate); return m, best

def predict_nn(m, P, idx, bs=16384):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; out.append(torch.sigmoid(m(torch.tensor(P.num(j), device=DEV), torch.tensor(P.cat(j), device=DEV))).cpu().numpy())
    return np.concatenate(out)

# ---------------- model zoo with equal random-search budget ----------------
def configs(name, rng):
    for _ in range(N_TRIALS):
        if name == 'LogReg': yield dict(C=float(10 ** rng.uniform(-3, 0)))
        elif name == 'LightGBM': yield dict(num_leaves=int(rng.choice([15, 31, 63])), learning_rate=float(rng.choice([0.02, 0.05])), min_child_samples=int(rng.choice([50, 200, 500])), n_estimators=int(rng.choice([200, 400])), reg_lambda=float(rng.choice([1, 5, 20])))
        elif name == 'XGBoost': yield dict(max_depth=int(rng.choice([4, 6, 8])), learning_rate=float(rng.choice([0.03, 0.08])), min_child_weight=float(rng.choice([5, 20, 50])), n_estimators=int(rng.choice([200, 400])), reg_lambda=float(rng.choice([1, 5, 20])))
        elif name == 'MLP': yield dict(h=int(rng.choice([64, 128, 256])), drop=float(rng.choice([0.1, 0.3])), lr=float(rng.choice([1e-3, 3e-3])))
        elif name == 'FTT': yield dict(dim=int(rng.choice([16, 32])), layers=int(rng.choice([1, 2])), lr=float(rng.choice([1e-3, 3e-3])))
        else: yield dict(ch=int(rng.choice([16, 32, 64])), hid=int(rng.choice([32, 64])), lr=float(rng.choice([1e-3, 3e-3])))

def fit_score(name, cfg, P, tri, vai, tei, perm, seed):
    """returns validation PR-AUC and test scores"""
    pw = W_NEG * (y[tri] == 0).sum() / max(1, (y[tri] == 1).sum())
    if name == 'LogReg':
        m = LogisticRegression(C=cfg['C'], class_weight={0: 1.0, 1: pw}, max_iter=400, tol=1e-3).fit(P.onehot(tri), y[tri])
        f = lambda idx: m.predict_proba(P.onehot(idx))[:, 1]
    elif name in ('LightGBM', 'XGBoost'):
        Z = lambda idx: np.hstack([np.nan_to_num(XN[idx], nan=-1), P.cat(idx)]).astype(np.float32)
        if name == 'LightGBM':
            m = lgb.LGBMClassifier(**cfg, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, scale_pos_weight=pw, n_jobs=4, verbose=-1, random_state=seed)
            m.fit(Z(tri), y[tri], categorical_feature=list(range(len(NUM), len(NUM) + len(CAT))))
        else:
            m = xgb.XGBClassifier(**cfg, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=pw, n_jobs=4, tree_method='hist', random_state=seed, eval_metric='aucpr')
            m.fit(Z(tri), y[tri])
        f = lambda idx: m.predict_proba(Z(idx))[:, 1]
    else:
        nn_, card = len(NUM), P.card(); lr = cfg['lr']
        if name == 'MLP': make = lambda: MLP(nn_, card, cfg['h'], cfg['drop'])
        elif name == 'FTT': make = lambda: FTT(nn_, card, cfg['dim'], cfg['layers'])
        else: make = lambda: CNNRNN(nn_, card, perm, ch=cfg['ch'], hid=cfg['hid'], attn=(name == 'AttnCNNRNN'))
        m, _ = train_nn(make, P, tri, vai, lr, seed=seed); f = lambda idx: predict_nn(m, P, idx)
    sv = f(vai); return average_precision_score(y[vai], sv, sample_weight=w_all[vai]), f, m

def metrics(yt, s, wt):
    prev = (yt * wt).sum() / wt.sum(); o = np.argsort(-s); cw = np.cumsum(wt[o]); r = {}
    r['prevalence'] = prev; r['pr_auc'] = average_precision_score(yt, s, sample_weight=wt); r['roc_auc'] = roc_auc_score(yt, s, sample_weight=wt)
    r['lift_pr'] = r['pr_auc'] / prev
    for k in (100, 500, 1000, 5000):                         # audit budgets in population providers
        sel = o[:np.searchsorted(cw, k) + 1]; tp = (yt[sel] * wt[sel]).sum()
        r[f'prec_at_{k}'] = tp / wt[sel].sum(); r[f'recall_at_{k}'] = tp / (yt * wt).sum()
    # raw calibration of the class-weighted scores (no prior correction)
    r['brier'] = brier_score_loss(yt, s, sample_weight=wt)
    bins = np.minimum((s * 10).astype(int), 9); ece = 0
    for b in range(10):
        mk = bins == b
        if mk.any(): ece += wt[mk].sum() / wt.sum() * abs((yt[mk] * wt[mk]).sum() / wt[mk].sum() - (s[mk] * wt[mk]).sum() / wt[mk].sum())
    r['ece'] = ece
    return r

done = set()
if os.path.exists(OUT):
    p = pd.read_csv(OUT); done = set(zip(p.design, p.rep, p.fold, p.model, p.perm))

def run_split(design, rep, fold, tr, te):
    rng = np.random.default_rng(1000 * rep + fold)
    # inner validation: 15% of training providers (grouped)
    vh = np.array([zlib.crc32((n + 'v' + str(rep)).encode()) % 100 for n in g[tr]])
    trn, va = tr[vh >= 15], tr[vh < 15]
    if DATA_FRAC < 1: trn = trn[np.array([zlib.crc32((n + 'lc').encode()) % 1000 for n in g[trn]]) < DATA_FRAC * 1000]
    neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
    vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
    P = Prep(tri)
    for name in MODELS:
        perms = [np.arange(len(NUM) + len(CAT))]
        if name in ('CNNRNN', 'AttnCNNRNN') and rep == 0:
            perms = perms + [np.random.default_rng(77 + k).permutation(len(NUM) + len(CAT)) for k in range(PERMS - 1)]
        for pi, perm in enumerate(perms):
            if (design, rep, fold, name, pi) in done: continue
            t0 = time.time(); best = (-1, None, None)
            for ci, cfg in enumerate(configs(name, np.random.default_rng(500 + 10 * fold + rep))):
                ap, fpred, _ = fit_score(name, cfg, P, tri, vai, te, perm, seed=ci)
                if ap > best[0]: best = (ap, fpred, cfg)
            best = (best[0], best[1](te), best[2])
            r = dict(dataset=DS, design=design, rep=rep, fold=fold, model=name, perm=pi, n_train=len(tri), pos_train=int(y[tri].sum()),
                     n_test=len(te), pos_test=int(y[te].sum()), val_pr_auc=best[0], cfg=json.dumps(best[2]), secs=round(time.time() - t0))
            r.update(metrics(y[te], best[1], w_all[te]))
            pd.DataFrame([r]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            np.save(f'{WORK}/scores/{DS}_{design}_{rep}_{fold}_{name}_{pi}' + (f'_lc{DATA_FRAC}' if DATA_FRAC < 1 else '') + '.npy', best[1].astype(np.float32))
            print(design, rep, fold, name, pi, 'PR', round(r['pr_auc'], 5), 'ROC', round(r['roc_auc'], 3), 'P@1000', round(r['prec_at_1000'], 4), r['secs'], 's', flush=True)

os.makedirs(f'{WORK}/scores', exist_ok=True)
if DESIGN == 'grouped':
    for rep in range(REPS):
        salt = np.array([zlib.crc32((n + str(rep)).encode()) for n in g]) % FOLDS
        for fold in range(FOLDS):
            te = np.where(salt == fold)[0]; np.save(f'{WORK}/scores/{DS}_grouped_{rep}_{fold}_test_idx.npy', te)
            run_split('grouped', rep, fold, np.where(salt != fold)[0], te)
else:
    trm = (yr >= 2013) & (yr <= 2019); tem = (yr >= 2020) & (yr <= 2022)
    pos_tr = set(g[trm & (y == 1)]); te = np.where(tem & ~pd.Series(g).isin(pos_tr).values)[0]
    np.save(f'{WORK}/scores/{DS}_temporal_0_0_test_idx.npy', te)
    for rep in range(REPS): run_split('temporal', rep, 0, np.where(trm)[0], te)
print('ALL DONE', flush=True)
