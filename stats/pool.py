"""Assemble the saved test scores of the five grouped folds of one repeat into one array per model (out-of-fold scores).
Usage (from the folder holding labels_<ds>.npz and scores/): python pool.py phys|partd <repeat>"""
import sys, numpy as np
ds, rep = sys.argv[1], int(sys.argv[2])
L = np.load(f'labels_{ds}.npz'); n = len(L['y'])
MODELS = ['LogReg','LightGBM','XGBoost','MLP','FTT','CNNRNN','AttnCNNRNN']
out = {}
for m in MODELS:
    s = np.full(n, np.nan, np.float32); cnt = 0
    for f in range(5):
        idx = np.load(f'scores/{ds}_grouped_{rep}_{f}_test_idx.npy'); sc = np.load(f'scores/{ds}_grouped_{rep}_{f}_{m}_0.npy')
        assert len(idx) == len(sc); s[idx] = sc; cnt += len(idx)
    assert cnt == n and not np.isnan(s).any()
    out[m] = s
np.savez(f'pooled_{ds}_{rep}.npz', **out); print(ds, rep, 'ok', n)
