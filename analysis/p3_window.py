"""Sensitivity to the label window: re-score saved grouped-design test scores keeping only positives 2-3 (or 3) years
before exclusion, i.e. dropping the year immediately before exclusion, when billing may already reflect an investigation.
Usage: python p3_window.py <results_dir> <dataset>  (needs _years_<ds>.parquet and _ytx_<ds>.parquet in results_dir)"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
RD, DS = sys.argv[1], sys.argv[2]
y_all = pd.read_parquet(f'{RD}/_years_{DS}.parquet').label.values.astype(int)
ytx = pd.read_parquet(f'{RD}/_ytx_{DS}.parquet').ytx.values
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']; rows = []
for win, keepset in [('1-3', {1, 2, 3}), ('2-3', {2, 3}), ('3', {3})]:
    for rep in range(3):
        for fold in range(5):
            key = f'{DS}_grouped_{rep}_{fold}'; fi = f'{RD}/scores/{key}_test_idx.npy'
            if not os.path.exists(fi): continue
            te = np.load(fi); yy = y_all[te]; keep = (yy == 0) | np.isin(ytx[te], list(keepset))
            y = yy[keep]; w = np.where(y == 1, 1.0, 4.0); prev = (y * w).sum() / w.sum()
            for m in M:
                fs = f'{RD}/scores/{key}_{m}_0.npy'
                if not os.path.exists(fs): continue
                s = np.load(fs)[keep]; o = np.argsort(-s); cw = np.cumsum(w[o]); i = min(np.searchsorted(cw, 1000), len(o) - 1)
                ap = average_precision_score(y, s, sample_weight=w)
                rows.append(dict(dataset=DS, window=win, rep=rep, fold=fold, model=m, pos_test=int(y.sum()), lift_pr=ap / prev,
                                 roc_auc=roc_auc_score(y, s, sample_weight=w), prec_at_1000=(y[o[:i + 1]] * w[o[:i + 1]]).sum() / cw[i]))
pd.DataFrame(rows).to_csv(f'{RD}/p3_window_{DS}.csv', index=False); print('done', DS, len(rows))
