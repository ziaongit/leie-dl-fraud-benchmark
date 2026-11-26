"""Paper 3 v2 - fold-averaged metrics with a provider-level bootstrap (complements the fold-level Nadeau-Bengio tests).
Variant used in the paper: metrics are computed within each of the 15 grouped test folds (as in Table 3) and averaged; the bootstrap
resamples providers (Poisson weights shared by all folds, repeats and models, so model differences are paired). Earlier note:
In each of the three grouped repeats every provider-year is scored exactly once (by the model of the fold in which it was a test row),
so the five test folds of a repeat together form one out-of-fold prediction for the whole sample (all ~1,450 Physician and
~2,200 Part D positive provider-years). Metrics are computed at population scale (negatives weight 4) and averaged over the
three repeats. Uncertainty: Poisson bootstrap over PROVIDERS (the same provider weights applied to all repeats and models, so
differences between models are paired), B replicates. Reports PR-AUC lift, precision among the top 1,000 and 5,000 and ROC-AUC,
with 95% percentile intervals, and paired differences LightGBM (or XGBoost) minus each other model.
Usage: python p3_pooled_boot.py <dir with labels_<ds>.npz and pooled_<ds>_<rep>.npz> <ds> [B]"""
import sys, time, numpy as np, pandas as pd
D, ds = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
MODELS = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']
L = np.load(f'{D}/labels_{ds}.npz'); y = L['y'].astype(np.float64); prov = L['prov']; base = np.where(y == 1, 1.0, 4.0)
F = np.load(f'{D}/folds_{ds}.npz')['F']; pre = {}; KEYS = [(r, f) for r in range(3) for f in range(5)]
for rep in range(3):
    P = np.load(f'{D}/pooled_{ds}_{rep}.npz')
    for m in MODELS:
        for f in range(5):
            ix = np.where(F[rep] == f)[0]; s = P[m][ix]; o = np.argsort(-s, kind='stable'); ss = s[o]
            last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1].astype(np.int64)
            pre[(rep, f, m)] = (prov[ix][o], y[ix][o], base[ix][o], last)
print(ds, 'prepared', flush=True)

def stats(pw, key):
    pv, yy, bb, last = pre[key]; w = bb * (pw[pv] if pw is not None else 1.0)
    tp = np.cumsum(w * yy); cw = np.cumsum(w); fp = cw - tp
    TP, FP = tp[last], fp[last]; P_, N_ = TP[-1], FP[-1]
    roc = np.trapezoid(np.r_[0, TP / P_], np.r_[0, FP / N_])
    prec = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
    ap = np.sum(np.diff(np.r_[0, TP / P_]) * prec); prev = P_ / (P_ + N_)
    r = [ap / prev, roc]
    for k in (1000, 5000):
        j = np.searchsorted(cw, k) + 1; r.append(tp[j - 1] / cw[j - 1])
    return np.array(r)

def draw(pw): return {m: np.mean([stats(pw, (r, f, m)) for r, f in KEYS], axis=0) for m in MODELS}
point = draw(None)
rng = np.random.default_rng(2027); nprov = prov.max() + 1; boot = {m: [] for m in MODELS}; t0 = time.time()
for b in range(B):
    pw = rng.poisson(1.0, nprov).astype(np.float64); d = draw(pw)
    for m in MODELS: boot[m].append(d[m])
    if b % 100 == 0: print(ds, b, round(time.time() - t0), 's', flush=True)
names = ['lift', 'roc', 'p1000', 'p5000']; rows = []
for m in MODELS:
    A = np.array(boot[m]); lo, hi = np.percentile(A, [2.5, 97.5], axis=0)
    r = dict(dataset=ds, model=m, B=B)
    for i, n in enumerate(names): r.update({n: point[m][i], n + '_lo': lo[i], n + '_hi': hi[i]})
    for ref in ('LightGBM', 'XGBoost'):
        dd = np.array(boot[ref]) - A; dlo, dhi = np.percentile(dd, [2.5, 97.5], axis=0)
        for i, n in enumerate(names):
            r[f'{ref}_minus_{n}'] = point[ref][i] - point[m][i]; r[f'{ref}_minus_{n}_lo'] = dlo[i]; r[f'{ref}_minus_{n}_hi'] = dhi[i]
            r[f'P_{ref}_better_{n}'] = float((dd[:, i] > 0).mean())
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{D}/p3_pooled_boot_{ds}.csv', index=False)
pd.set_option('display.width', 250)
print(R[['model', 'lift', 'lift_lo', 'lift_hi', 'p1000', 'p1000_lo', 'p1000_hi', 'LightGBM_minus_p1000_lo', 'LightGBM_minus_p1000_hi', 'P_LightGBM_better_p1000', 'LightGBM_minus_lift_lo', 'LightGBM_minus_lift_hi', 'P_LightGBM_better_lift']].round(4).to_string())
