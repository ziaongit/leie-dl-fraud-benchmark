"""Paper 3 v2 - provider-level (cluster) bootstrap confidence intervals for the 32-state independent-label test (run 7).
Pairs the saved run-7 test scores with the rebuilt test rows (rebuild_test.py, identical order) and resamples PROVIDERS
(Poisson bootstrap: each provider gets a Poisson(1) weight, applied to all its provider-years), B = 2000.
Reports ROC-AUC, PR-AUC lift and precision among the top 1,000 (population scale, row weight 4) with 95% percentile intervals,
and paired differences LightGBM minus each other model on the same bootstrap draws."""
import sys, numpy as np, pandas as pd
D = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MODELS = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']

def prep(s):
    o = np.argsort(-s, kind='stable'); ss = s[o]
    return o, np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]   # order and last index of each tie group

def curve_stats(s, y, w, k=1000, pre=None):
    o, last = pre if pre is not None else prep(s); y, w = y[o], w[o]
    tp = np.cumsum(w * y); fp = np.cumsum(w * (1 - y))
    TP, FP = tp[last], fp[last]; P, N = TP[-1], FP[-1]
    tpr = np.r_[0, TP / P]; fpr = np.r_[0, FP / N]
    roc = np.trapezoid(tpr, fpr)
    prec = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0); rec = TP / P
    ap = np.sum(np.diff(np.r_[0, rec]) * prec)
    prev = P / (P + N)
    cw = np.cumsum(4 * w); j = np.searchsorted(cw, k) + 1      # population scale: each row stands for 4 provider-years
    p_at_k = (w[:j] * y[:j]).sum() / max(w[:j].sum(), 1e-12)
    return roc, ap / prev, p_at_k

rows = []
for ds in ['phys', 'partd']:
    t = np.load(f'{D}/test_{ds}.npz'); y = t['yt'].astype(float); prov = t['prov']
    S = {m: np.load(f'{D}/scores/states_{ds}_{m}.npy').astype(np.float64) for m in MODELS}
    for m in MODELS: assert len(S[m]) == len(y)
    PRE = {m: prep(S[m]) for m in MODELS}
    point = {m: curve_stats(S[m], y, np.ones_like(y), pre=PRE[m]) for m in MODELS}
    rng = np.random.default_rng(2026); nprov = prov.max() + 1
    boot = {m: [] for m in MODELS}
    for b in range(B):
        wp = rng.poisson(1.0, nprov).astype(float); w = wp[prov]
        if (w * y).sum() == 0: continue
        for m in MODELS: boot[m].append(curve_stats(S[m], y, w, pre=PRE[m]))
    for m in MODELS:
        A = np.array(boot[m]); lo, hi = np.percentile(A, [2.5, 97.5], axis=0); L = np.array(boot['LightGBM'])
        d = L - A; dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
        rows.append(dict(dataset=ds, model=m, positives=int(y.sum()), pos_providers=len(set(prov[y == 1])), B=len(A),
                         roc=point[m][0], roc_lo=lo[0], roc_hi=hi[0], lift=point[m][1], lift_lo=lo[1], lift_hi=hi[1],
                         p1000=point[m][2], p1000_lo=lo[2], p1000_hi=hi[2],
                         lgbm_minus_roc_lo=dlo[0], lgbm_minus_roc_hi=dhi[0], lgbm_minus_lift_lo=dlo[1], lgbm_minus_lift_hi=dhi[1],
                         p_lgbm_lift_not_better=float((d[:, 1] <= 0).mean())))
    print(ds, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{D}/p3_states_ci.csv', index=False)
pd.set_option('display.width', 250)
print(R[['dataset', 'model', 'roc', 'roc_lo', 'roc_hi', 'lift', 'lift_lo', 'lift_hi', 'p1000', 'lgbm_minus_lift_lo', 'lgbm_minus_lift_hi', 'p_lgbm_lift_not_better']].round(3).to_string())
