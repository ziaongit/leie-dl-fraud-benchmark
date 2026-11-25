"""Paper 3 v2 - Are attention weights faithful feature explanations? (answers IJDSA R2-9)
For the attention CNN-RNN trained on confirmed (LEIE) labels, compare per-feature attention with
 (a) SHAP (GradientExplainer on the same network),
 (b) permutation importance (drop in validation PR-AUC when a feature is shuffled),
 (c) ablation (drop in PR-AUC when a feature is replaced by its training mean / 'other' category),
under the original feature order and under random feature orders (attention is mapped back to features).
Also reports the same SHAP-vs-permutation agreement for LightGBM (TreeSHAP) as a reference.
Usage on Kaggle: python p3_attention.py phys   (folds 0-2 of repeat 0, 3 feature orders)"""
import os, sys, glob, zlib, json, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import lightgbm as lgb, shap, warnings
warnings.filterwarnings('ignore')
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else 'phys', 'grouped']
os.environ.setdefault('MODELS', 'AttnCNNRNN')
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p3_bench.py')).read()
src = src[:src.index("done = set()")]                      # reuse data loading, Prep, models, train_nn, predict_nn, configs
exec(src)
OUTA = f'{WORK}/p3_attention_results_{DS}.csv'; FEATS = NUM + CAT; F = len(FEATS)
rows, imp_rows = [], []

def pr(idx, s): return average_precision_score(y[idx], s, sample_weight=w_all[idx])

for fold in range(int(os.environ.get('ATT_FOLDS', '3'))):
    rep = 0; salt = np.array([zlib.crc32((n + str(rep)).encode()) for n in g]) % FOLDS
    tr, te = np.where(salt != fold)[0], np.where(salt == fold)[0]
    rng = np.random.default_rng(1000 * rep + fold)
    vh = np.array([zlib.crc32((n + 'v' + str(rep)).encode()) % 100 for n in g[tr]])
    trn, va = tr[vh >= 15], tr[vh < 15]
    neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
    vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
    P = Prep(tri)
    # evaluation subset: all test positives + 20k negatives (weighted) to keep permutation/SHAP affordable
    tneg = te[y[te] == 0]; ev = np.concatenate([te[y[te] == 1], rng.choice(tneg, size=min(len(tneg), 20000), replace=False)])
    wscale = np.where(y[ev] == 1, 1.0, W_NEG * (len(tneg) / min(len(tneg), 20000)))
    prw = lambda s: average_precision_score(y[ev], s, sample_weight=wscale)
    for pi in range(PERMS):
        perm = np.arange(F) if pi == 0 else np.random.default_rng(77 + pi).permutation(F)
        cfg = dict(ch=32, hid=64, lr=1e-3)
        m, _ = train_nn(lambda: CNNRNN(len(NUM), P.card(), perm, ch=cfg['ch'], hid=cfg['hid'], attn=True), P, tri, vai, cfg['lr'], seed=pi)
        m.eval()
        Xn_ev, Xc_ev = torch.tensor(P.num(ev), device=DEV), torch.tensor(P.cat(ev), device=DEV)
        with torch.no_grad():
            s0 = torch.sigmoid(m(Xn_ev, Xc_ev)).cpu().numpy(); A = m.last_attn.cpu().numpy()   # (n, F) over permuted positions
        att = np.zeros(F); att[perm] = A[y[ev] == 1].mean(0)          # map positions back to features (fraud cases)
        base = prw(s0)
        # permutation importance and ablation (feature-level)
        permI, ablI = np.zeros(F), np.zeros(F)
        for j in range(F):
            for _r in range(3):
                Xn2, Xc2 = Xn_ev.clone(), Xc_ev.clone(); idx = torch.randperm(len(ev), device=DEV)
                if j < len(NUM): Xn2[:, j] = Xn2[idx, j]
                else: Xc2[:, j - len(NUM)] = Xc2[idx, j - len(NUM)]
                with torch.no_grad(): permI[j] += (base - prw(torch.sigmoid(m(Xn2, Xc2)).cpu().numpy())) / 3
            Xn3, Xc3 = Xn_ev.clone(), Xc_ev.clone()
            if j < len(NUM): Xn3[:, j] = 0.0                          # standardised mean
            else: Xc3[:, j - len(NUM)] = 0                            # 'other/unseen' category
            with torch.no_grad(): ablI[j] = base - prw(torch.sigmoid(m(Xn3, Xc3)).cpu().numpy())
        # SHAP on the network (gradient explainer on numeric inputs; categorical embeddings held fixed)
        posi = np.where(y[ev] == 1)[0][:300]; bg = np.random.default_rng(0).choice(np.where(y[ev] == 0)[0], 200, replace=False)
        class Wrap(torch.nn.Module):
            def __init__(s, mm, xc): super().__init__(); s.mm, s.xc = mm, xc
            def forward(s, xn): return s.mm(xn, s.xc[:len(xn)]).unsqueeze(1)
        try:
            with torch.backends.cudnn.flags(enabled=False):
                ex = shap.GradientExplainer(Wrap(m, Xc_ev[posi]), Xn_ev[bg]); sv = ex.shap_values(Xn_ev[posi])
            sv = np.abs(np.array(sv).reshape(len(posi), -1)).mean(0)
            shapI = np.concatenate([sv, np.full(len(CAT), np.nan)])
        except Exception as e:
            print('shap failed', e, flush=True); shapI = np.full(F, np.nan)
        num = slice(0, len(NUM))
        def rho(a, b):
            mk = ~np.isnan(a) & ~np.isnan(b); return spearmanr(a[mk], b[mk]).correlation if mk.sum() > 3 else np.nan
        def top10(a, b):
            mk = ~np.isnan(a) & ~np.isnan(b)
            if mk.sum() < 10: return np.nan
            ia, ib = set(np.argsort(-np.where(mk, a, -9e9))[:10]), set(np.argsort(-np.where(mk, b, -9e9))[:10]); return len(ia & ib) / len(ia | ib)
        r = dict(dataset=DS, fold=fold, perm=pi, pr_auc_eval=base,
                 rho_att_perm=rho(att, permI), rho_att_abl=rho(att, ablI), rho_att_shap=rho(att[num], shapI[num]),
                 rho_shap_perm=rho(shapI[num], permI[num]), rho_perm_abl=rho(permI, ablI),
                 top10_att_perm=top10(att, permI), top10_shap_perm=top10(shapI, permI),
                 att_position_corr=spearmanr(np.arange(F), A[y[ev] == 1].mean(0)).correlation)   # does attention just follow position?
        rows.append(r); print(r, flush=True)
        for j, fn in enumerate(FEATS):
            imp_rows.append(dict(dataset=DS, fold=fold, perm=pi, feature=fn, attention=att[j], permutation=permI[j], ablation=ablI[j], shap=shapI[j]))
        pd.DataFrame(rows).to_csv(OUTA, index=False); pd.DataFrame(imp_rows).to_csv(f'{WORK}/p3_attention_importances_{DS}.csv', index=False)
    # reference: LightGBM TreeSHAP vs permutation on the same fold
    Z = lambda idx: np.hstack([np.nan_to_num(XN[idx], nan=-1), P.cat(idx)]).astype(np.float32)
    pw = W_NEG * (y[tri] == 0).sum() / max(1, (y[tri] == 1).sum())
    lm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, reg_lambda=5, scale_pos_weight=pw, n_jobs=4, verbose=-1).fit(
        Z(tri), y[tri], categorical_feature=list(range(len(NUM), F)))
    Ze = Z(ev); s0 = lm.predict_proba(Ze)[:, 1]; base = prw(s0); permI = np.zeros(F)
    for j in range(F):
        for _r in range(3):
            Z2 = Ze.copy(); Z2[:, j] = Z2[np.random.default_rng(100 * j + _r).permutation(len(ev)), j]; permI[j] += (base - prw(lm.predict_proba(Z2)[:, 1])) / 3
    tsv = np.abs(shap.TreeExplainer(lm).shap_values(Ze[y[ev] == 1][:300])); tsv = tsv[1] if isinstance(tsv, list) else tsv; tsv = tsv.mean(0)
    rows.append(dict(dataset=DS, fold=fold, perm=-1, model='LightGBM', pr_auc_eval=base, rho_shap_perm=spearmanr(tsv, permI).correlation,
                     top10_shap_perm=len(set(np.argsort(-tsv)[:10]) & set(np.argsort(-permI)[:10])) / len(set(np.argsort(-tsv)[:10]) | set(np.argsort(-permI)[:10]))))
    pd.DataFrame(rows).to_csv(OUTA, index=False); print(rows[-1], flush=True)
print('ATTENTION DONE', flush=True)
