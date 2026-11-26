"""Recalibration demonstration: on each grouped test fold, providers are split in two halves (hash of row index group via
test position parity of a provider hash); a recalibrator is fitted on one half and applied to the other (cross-fitting),
as a payer would fit it on a labelled hold-out set at natural prevalence. Methods: Platt scaling on the logit of the score
and isotonic regression (both weighted, negatives x4). Metrics: Brier score and expected calibration error with 20
equal-mass bins (weighted), before and after.  Usage: python p3_recal.py <results_dir> <dataset>"""
import sys, numpy as np, pandas as pd, zlib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
RD, DS = sys.argv[1], sys.argv[2]
y_all = pd.read_parquet(f'{RD}/_years_{DS}.parquet').label.values.astype(int)
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']; rows = []
def ece(y, p, w, nb=20):
    o = np.argsort(p); cw = np.cumsum(w[o]); edges = np.searchsorted(cw, np.linspace(0, cw[-1], nb + 1)[1:-1]); e = 0
    for b in np.split(o, edges):
        if len(b): e += w[b].sum() / w.sum() * abs((y[b] * w[b]).sum() / w[b].sum() - (p[b] * w[b]).sum() / w[b].sum())
    return e
def brier(y, p, w): return (w * (p - y) ** 2).sum() / w.sum()
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
import os
REPS=[int(r) for r in os.environ.get('REPS','0,1,2').split(',')]
for rep in REPS:
    for fold in range(5):
        key = f'{DS}_grouped_{rep}_{fold}'; te = np.load(f'{RD}/scores/{key}_test_idx.npy'); y = y_all[te]; w = np.where(y == 1, 1.0, 4.0)
        half = (te % 2 == 0)   # rows split by parity of their index (fixed, independent of labels and scores)
        for m in M:
            s = np.load(f'{RD}/scores/{key}_{m}_0.npy').astype(float); pi = np.zeros_like(s); pp = np.zeros_like(s)
            for a, b in [(half, ~half), (~half, half)]:
                iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(s[a], y[a], sample_weight=w[a]); pi[b] = iso.predict(s[b])
                pl = LogisticRegression(C=1e4, max_iter=500).fit(lg(s[a]).reshape(-1, 1), y[a], sample_weight=w[a]); pp[b] = pl.predict_proba(lg(s[b]).reshape(-1, 1))[:, 1]
            prev = (y * w).sum() / w.sum()
            rows.append(dict(dataset=DS, rep=rep, fold=fold, model=m, prevalence=prev,
                             ece_raw=ece(y, s, w), ece_platt=ece(y, pp, w), ece_iso=ece(y, pi, w),
                             brier_raw=brier(y, s, w), brier_platt=brier(y, pp, w), brier_iso=brier(y, pi, w),
                             mean_pred_raw=(s * w).sum() / w.sum(), mean_pred_platt=(pp * w).sum() / w.sum()))
R = pd.DataFrame(rows); R.to_csv(f'{RD}/p3_recal_{DS}' + ('' if len(REPS) == 3 else '_' + '_'.join(map(str, REPS))) + '.csv', index=False)
print(R.groupby('model')[['prevalence', 'mean_pred_raw', 'mean_pred_platt', 'ece_raw', 'ece_platt', 'ece_iso', 'brier_raw', 'brier_platt', 'brier_iso']].mean().to_string(float_format=lambda v: f'{v:.2e}'))
