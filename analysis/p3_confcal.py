"""Confusion counts at audit budgets and reliability-diagram data from saved run-1 test scores (perm 0, grouped design).
Counts are scaled to the full population (negatives carry weight 4). Usage: python p3_confcal.py <results_dir> <dataset>"""
import sys, glob, os, numpy as np, pandas as pd
RD, DS = sys.argv[1], sys.argv[2]; L = np.load(f'{RD}/test_labels.npz')
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']
bins = np.r_[0, np.logspace(-5, 0, 11)]
conf, cal = [], []
for rep in range(3):
    for fold in range(5):
        key = f'{DS}_grouped_{rep}_{fold}'
        if key + '_y' not in L: continue
        y, w = L[key + '_y'], L[key + '_w']; P = (y * w).sum()
        for m in M:
            f = f'{RD}/scores/{key}_{m}_0.npy'
            if not os.path.exists(f): continue
            s = np.load(f).astype(float); o = np.argsort(-s); cw = np.cumsum(w[o]); cp = np.cumsum((y * w)[o])
            for k in [1000, 5000]:
                i = min(np.searchsorted(cw, k), len(o) - 1); tp = cp[i]; flagged = cw[i]
                conf.append(dict(dataset=DS, rep=rep, fold=fold, model=m, k=k, TP=tp, FP=flagged - tp, FN=P - tp, positives=P))
            b = np.clip(np.digitize(s, bins) - 1, 0, len(bins) - 2)
            for j in range(len(bins) - 1):
                mk = b == j
                if mk.sum(): cal.append(dict(dataset=DS, rep=rep, fold=fold, model=m, bin=j, w=w[mk].sum(), pred=(s[mk] * w[mk]).sum(), obs=(y[mk] * w[mk]).sum()))
pd.DataFrame(conf).to_csv(f'{RD}/p3_confusion_{DS}.csv', index=False); pd.DataFrame(cal).to_csv(f'{RD}/p3_calibration_{DS}.csv', index=False); print('done', DS, len(conf))
