"""Rebuild the run-7 test rows (same order as p3_states.py) so that saved scores can be paired with labels."""
import sys, glob, zlib, numpy as np, pandas as pd
SDS = sys.argv[1]; base = sys.argv[2]
if SDS == 'phys':
    files = sorted(glob.glob(f'{base}/phys/provider_year_*.parquet')); ID = 'Rndrng_NPI'
else:
    files = sorted(glob.glob(f'{base}/partd/partd_year_*.parquet')); ID = 'NPI'
parts = []
for f in files:
    t = pd.read_parquet(f, columns=[ID, 'year', 'label'])
    if SDS == 'phys': t = t[(t.label == 1) | t[ID].map(lambda n: zlib.crc32(n.encode()) % 4 == 0)]
    parts.append(t)
d = pd.concat(parts, ignore_index=True)
g = d[ID].values; yr = d.year.values.astype(int); y = d.label.values.astype(int)
st = pd.read_csv(f'{base}/run7/state_medicaid_exclusions_npi_20261126.csv', dtype=str); st = st[st.start_date.notna()]
st['ey'] = st.start_date.str[:4].astype(int)
st = st[~st.NPI.isin(set(g[y == 1]))]
first = st.sort_values('ey').drop_duplicates('NPI').set_index('NPI')
on = pd.Series(g).isin(set(first.index)).values
ey = pd.Series(g).map(first.ey).values.astype(float); stt = pd.Series(g).map(first.state).values
pos = on & ((ey - yr) >= 1) & ((ey - yr) <= 3)
idx = np.where((yr >= 2020) & (yr <= 2022) & (y == 0) & (~on | pos))[0]
yt = pos[idx].astype(np.int8)
codes, uniq = pd.factorize(g[idx])
np.savez_compressed(f'{base}/run7/ci/test_{SDS}.npz', yt=yt, prov=codes.astype(np.int32), state=np.where(pd.isna(stt[idx]), '', stt[idx]).astype('U2'), year=yr[idx].astype(np.int16))
print(SDS, 'rows', len(d), 'test', len(idx), 'pos', int(yt.sum()), 'providers', len(set(g[idx][yt == 1])))
