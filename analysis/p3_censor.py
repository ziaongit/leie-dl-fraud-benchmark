"""Sensitivity to right-censored labels: recompute grouped-design metrics on saved test scores after removing test
provider-years from data years with an incomplete 3-year look-ahead in the LEIE (default: years >= 2023).
Usage: python p3_censor.py <results_dir> <data_root> <dataset> [max_year]   (needs <results_dir>/_years_<dataset>.parquet with columns year, label in benchmark loading order -
  build it by concatenating data/phys or data/partd files sorted by name; env REPS)"""
import sys, glob, os, zlib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
RD, DATA, DS = sys.argv[1], sys.argv[2], sys.argv[3]; MAXY = int(sys.argv[4]) if len(sys.argv) > 4 else 2022
d = pd.read_parquet(f'{RD}/_years_{DS}.parquet')   # year and label of every row, in benchmark loading order (see p3_labels.py)
y_all = d.label.values.astype(int); yr_all = d.year.values.astype(int)
REPS = [int(r) for r in os.environ.get('REPS', '0,1,2').split(',')]
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']; rows = []
for rep in REPS:
    for fold in range(5):
        key = f'{DS}_grouped_{rep}_{fold}'; fi = f'{RD}/scores/{key}_test_idx.npy'
        if not os.path.exists(fi): continue
        te = np.load(fi); keep = yr_all[te] <= MAXY; y = y_all[te][keep]; w = np.where(y == 1, 1.0, 4.0); prev = (y * w).sum() / w.sum()
        for m in M:
            fs = f'{RD}/scores/{key}_{m}_0.npy'
            if not os.path.exists(fs): continue
            s = np.load(fs)[keep]; o = np.argsort(-s); cw = np.cumsum(w[o]); i = min(np.searchsorted(cw, 1000), len(o) - 1)
            ap = average_precision_score(y, s, sample_weight=w)
            rows.append(dict(dataset=DS, rep=rep, fold=fold, model=m, max_year=MAXY, pos_test=int(y.sum()), prevalence=prev, pr_auc=ap, lift_pr=ap / prev,
                             roc_auc=roc_auc_score(y, s, sample_weight=w), prec_at_1000=(y[o[:i + 1]] * w[o[:i + 1]]).sum() / cw[i]))
pd.DataFrame(rows).to_csv(f'{RD}/p3_censor_{DS}_' + '_'.join(map(str, REPS)) + '.csv', index=False); print('done', DS, len(rows))
