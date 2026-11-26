"""Rebuild label and provider arrays in the row order of benchmark/p3_bench.py (needed by stats/pool.py and stats/p3_pooled_boot.py).
Usage (from the folder holding phys/ and partd/ parquet files): python build_labels.py phys|partd"""
import sys, glob, zlib, numpy as np, pandas as pd
SDS=sys.argv[1]
files=sorted(glob.glob(f'{SDS}/provider_year_*.parquet' if SDS=='phys' else f'{SDS}/partd_year_*.parquet')); ID='Rndrng_NPI' if SDS=='phys' else 'NPI'
parts=[]
for f in files:
    t=pd.read_parquet(f,columns=[ID,'label'])
    if SDS=='phys': t=t[(t.label==1)|t[ID].map(lambda n: zlib.crc32(n.encode())%4==0)]
    parts.append(t)
d=pd.concat(parts,ignore_index=True); codes,_=pd.factorize(d[ID])
np.savez_compressed(f'run1/labels_{SDS}.npz', y=d.label.values.astype(np.int8), prov=codes.astype(np.int32))
print(SDS,len(d),int(d.label.sum()))
