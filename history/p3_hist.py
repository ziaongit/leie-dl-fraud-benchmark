"""Paper 3 v2 - sequence models on GENUINE sequences: each provider's own billing history over years.
For every target provider-year t (Physician data, the benchmark sample), the input is the provider's numeric profile in
years t-3, t-2, t-1 and t (missing years zero-filled with a presence flag); categorical variables are those of year t.
Earlier years of fraud providers that are not in the benchmark sample (more than three years before exclusion) are
added from hist_extra_rows_phys.parquet so that fraud and non-fraud providers have comparable histories; their labels are
never used. Same folds, training subset, preprocessing (fitted on training target rows), tuning budget and metrics as
p3_bench.py (grouped design, repeat 0, five folds).
Models: LightGBM on year t only (reference), LightGBM and MLP on the flattened 4-year history (order-free use of the same
information), GRU over years and GRU with additive attention over years (sequence models where order is meaningful).
Usage: python p3_hist.py   (env: WORK, HMODELS, HFOLDS)"""
import os, sys, glob, json, time, zlib
sys.argv = [sys.argv[0], 'phys', 'grouped', '1', '5']
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, 'p3_bench.py')] + glob.glob('/kaggle/input/**/p3_bench.py', recursive=True)
src = open([c for c in cands if os.path.exists(c)][0]).read()
src = src.replace("NCAT = [len(X[c].cat.categories) for c in CAT]; del X", "NCAT = [len(X[c].cat.categories) for c in CAT]; CATS = {c: list(X[c].cat.categories) for c in CAT}; del X")
src = src[:src.index("done = set()")]
exec(src)
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
WORKH = os.environ.get('WORK', '/kaggle/working/hist'); os.makedirs(WORKH, exist_ok=True)
OUTH = f'{WORKH}/results_hist_phys.csv'
HMODELS = os.environ.get('HMODELS', 'LightGBM_cur,LightGBM_hist,MLP_hist,GRU_hist,AttnGRU_hist').split(',')
L = 4                                                          # years t-3 .. t

# ---- append history-only rows (earlier years of fraud providers) ----
xf = glob.glob('/kaggle/input/**/hist_extra_rows_phys.parquet', recursive=True) + glob.glob(os.path.join(os.environ.get('DATA', '.'), '**/hist_extra_rows_phys.parquet'), recursive=True)
x = pd.read_parquet([f for f in xf if os.path.exists(f)][0])
n0 = len(y)
XN = np.vstack([XN, x[NUM].values.astype(np.float32)])
XC = np.vstack([XC, np.stack([pd.Categorical(x[c].astype(str), categories=CATS[c]).codes for c in CAT], 1).astype(np.int32)])
g_all = np.concatenate([g, x.Rndrng_NPI.values]); yr_all = np.concatenate([yr, x.year.values.astype(int)])
key = pd.Series(np.arange(len(g_all)), index=pd.MultiIndex.from_arrays([g_all, yr_all]))
key = key[~key.index.duplicated()]
H = np.full((n0, L), -1, np.int64); H[:, L - 1] = np.arange(n0)
for k in range(1, L):
    H[:, L - 1 - k] = key.reindex(pd.MultiIndex.from_arrays([g, yr - k])).fillna(-1).values.astype(np.int64)
print('history present (lag1, lag2, lag3):', [(H[:, L - 1 - k] >= 0).mean().round(3) for k in (1, 2, 3)], 'extra rows', len(x), flush=True)
F = len(NUM)

class HistP:                                                   # history features on top of Prep (fitted on training target rows)
    def __init__(self, P): self.P = P
    def num(self, idx):
        out = np.zeros((len(idx), L, F + 1), np.float32)
        for s in range(L):
            h = H[idx, s]; ok = h >= 0
            if ok.any(): out[ok, s, :F] = self.P.num(h[ok]); out[ok, s, F] = 1.0
        return out.reshape(len(idx), L * (F + 1))
    def cat(self, idx): return self.P.cat(idx)
    def card(self): return self.P.card()

class HistGRU(nn.Module):
    def __init__(s, card, hid=64, attn=False, emb=8, drop=0.2):
        super().__init__(); s.inp = nn.Linear(F + 1, hid); s.gru = nn.GRU(hid, hid, batch_first=True); s.attn = attn
        s.E = nn.ModuleList([nn.Embedding(k, emb) for k in card])
        if attn: s.a = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1, bias=False))
        s.h = nn.Sequential(nn.Dropout(drop), nn.Linear(hid + emb * len(card), 1))
    def forward(s, xn, xc):
        z, _ = s.gru(torch.relu(s.inp(xn.view(len(xn), L, F + 1))))
        if s.attn: a = torch.softmax(s.a(z).squeeze(2), 1); z = (a.unsqueeze(2) * z).sum(1)
        else: z = z[:, -1]
        return s.h(torch.cat([z] + [e(xc[:, j]) for j, e in enumerate(s.E)], 1)).squeeze(1)

def hconfigs(name, rng):
    for _ in range(N_TRIALS):
        if name.startswith('LightGBM'): yield dict(num_leaves=int(rng.choice([15, 31, 63])), learning_rate=float(rng.choice([0.02, 0.05])), min_child_samples=int(rng.choice([50, 200, 500])), n_estimators=int(rng.choice([200, 400])), reg_lambda=float(rng.choice([1, 5, 20])))
        elif name == 'MLP_hist': yield dict(h=int(rng.choice([64, 128, 256])), drop=float(rng.choice([0.1, 0.3])), lr=float(rng.choice([1e-3, 3e-3])))
        else: yield dict(hid=int(rng.choice([32, 64, 128])), lr=float(rng.choice([1e-3, 3e-3])))

def hfit(name, cfg, P, HP, tri, vai, seed):
    pw = W_NEG * (y[tri] == 0).sum() / max(1, (y[tri] == 1).sum())
    if name.startswith('LightGBM'):
        if name == 'LightGBM_cur': Z = lambda idx: np.hstack([np.nan_to_num(XN[idx], nan=-1), P.cat(idx)]).astype(np.float32); nnum = F
        else: Z = lambda idx: np.hstack([HP.num(idx), HP.cat(idx)]).astype(np.float32); nnum = L * (F + 1)
        m = lgb.LGBMClassifier(**cfg, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, scale_pos_weight=pw, n_jobs=4, verbose=-1, random_state=seed)
        m.fit(Z(tri), y[tri], categorical_feature=list(range(nnum, nnum + len(CAT)))); f = lambda idx: m.predict_proba(Z(idx))[:, 1]
    else:
        if name == 'MLP_hist': make = lambda: MLP(L * (F + 1), HP.card(), cfg['h'], cfg['drop'])
        else: make = lambda: HistGRU(HP.card(), cfg['hid'], attn=(name == 'AttnGRU_hist'))
        m, _ = train_nn(make, HP, tri, vai, cfg['lr'], seed=seed); f = lambda idx: predict_nn(m, HP, idx)
    return average_precision_score(y[vai], f(vai), sample_weight=w_all[vai]), f

done = set(pd.read_csv(OUTH).key) if os.path.exists(OUTH) else set()
rep = 0; salt = np.array([zlib.crc32((n + str(rep)).encode()) for n in g]) % FOLDS
for fold in range(int(os.environ.get('HFOLDS', FOLDS))):
    tr, te = np.where(salt != fold)[0], np.where(salt == fold)[0]
    rng = np.random.default_rng(1000 * rep + fold)
    vh = np.array([zlib.crc32((n + 'v' + str(rep)).encode()) % 100 for n in g[tr]]); trn, va = tr[vh >= 15], tr[vh < 15]
    neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
    vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
    P = Prep(tri); HP = HistP(P)
    for name in HMODELS:
        k = f'hist_{fold}_{name}'
        if k in done: continue
        t0 = time.time(); best = (-1, None, None)
        for ci, cfg in enumerate(hconfigs(name, np.random.default_rng(500 + 10 * fold + rep))):
            ap, f = hfit(name, cfg, P, HP, tri, vai, seed=ci)
            if ap > best[0]: best = (ap, f, cfg)
        s = best[1](te); r = dict(key=k, fold=fold, model=name, val_pr_auc=best[0], cfg=json.dumps(best[2]), secs=round(time.time() - t0), pos_test=int(y[te].sum()))
        r.update(metrics(y[te], s, w_all[te])); pd.DataFrame([r]).to_csv(OUTH, mode='a', header=not os.path.exists(OUTH), index=False)
        np.save(f'{WORKH}/hist_{fold}_{name}.npy', s.astype(np.float32))
        print('hist', fold, name, 'PR', round(r['pr_auc'], 5), 'lift', round(r['lift_pr'], 1), 'ROC', round(r['roc_auc'], 3), 'P@1000', round(r['prec_at_1000'], 4), r['secs'], 's', flush=True)
print('HIST DONE', flush=True)
