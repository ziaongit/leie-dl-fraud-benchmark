"""Step 3: provider-level sampling of negatives for the Physician data (as used in the paper and on Kaggle).
Keeps every positive provider-year and all years of never-excluded providers with crc32(NPI) % 4 == 0 (these carry weight 4
in every metric), and drops beneficiary race variables. Usage: python data_prep/03_sample_physician.py <analysis_dir> <out_dir>"""
import sys, os, glob, zlib, pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f'{src}/provider_year_*.parquet')):
    t = pd.read_parquet(f); t = t.drop(columns=[c for c in t.columns if c.startswith('Bene_Race_')])
    t = t[(t.label == 1) | t.Rndrng_NPI.map(lambda n: zlib.crc32(n.encode()) % 4 == 0)]
    t.to_parquet(f'{out}/{os.path.basename(f)}', index=False); print(os.path.basename(f), len(t), int(t.label.sum()))
