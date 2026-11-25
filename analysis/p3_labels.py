"""Rebuild test labels/weights for every saved test index (same data loading order as p3_bench.py) -> test_labels.npz
Usage: python p3_labels.py <results_dir> <data_root>"""
import sys, glob, os, zlib, numpy as np, pandas as pd
RD, DATA = sys.argv[1], sys.argv[2]; out = {}
for ds, pat, ID in [('phys', 'provider_year_*.parquet', 'Rndrng_NPI'), ('partd', 'partd_year_*.parquet', 'NPI')]:
    files = sorted(glob.glob(f'{DATA}/**/{pat}', recursive=True))
    if not files or not glob.glob(f'{RD}/scores/{ds}_*_test_idx.npy'): continue
    parts = []
    for f in files:
        t = pd.read_parquet(f, columns=[ID, 'label'])
        if ds == 'phys': t = t[(t.label == 1) | t[ID].map(lambda n: zlib.crc32(n.encode()) % 4 == 0)]
        parts.append(t)
    y = pd.concat(parts, ignore_index=True).label.values.astype(np.int8); w = np.where(y == 1, 1.0, 4.0)
    for f in glob.glob(f'{RD}/scores/{ds}_*_test_idx.npy'):
        key = os.path.basename(f).replace('_test_idx.npy', ''); te = np.load(f); out[key + '_y'] = y[te]; out[key + '_w'] = w[te]
np.savez_compressed(f'{RD}/test_labels.npz', **out); print(len(out) // 2, 'splits')
