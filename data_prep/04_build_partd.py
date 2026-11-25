"""Step 4: Part D dataset - CMS Medicare Part D Prescribers by Provider.
Builds one analysis parquet per data year with the SAME LEIE label rule as steps 01/02:
  label = 1 : provider-year 1-3 years before a fraud-related LEIE exclusion (1128a1/a3/b7, valid NPI)
  label = 0 : provider never appears in the LEIE (any type, any date)
  dropped   : provider-years in/after exclusion, other LEIE types, fraud exclusions > 3 years later
Features: prescribing volume, cost, brand/generic, opioid/antibiotic/antipsychotic, plan-type and beneficiary mix.
Names, addresses and race variables are dropped. Nothing is derived from LEIE.
Usage: python data_prep/04_build_partd.py <folder with Part D by-provider CSVs> <leie_csv> <out_dir> [year]\nNegatives are sampled at provider level (crc32(NPI) % 4 == 0, weight 4), exactly as in data_prep/03.
Data year is read from the file name (..._DYY_...) or from a 4-digit year in the name."""
import pandas as pd, numpy as np, sys, os, re, glob, zlib
src, leie_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

L = pd.read_csv(leie_path, dtype=str, keep_default_na=False)
L['NPI'] = L['NPI'].str.strip(); L = L[L['NPI'].str.fullmatch(r'[1-9]\d{9}')].copy()
L['excl_type'] = L['EXCLTYPE'].str.strip().str.lower()
L['excl_year'] = pd.to_datetime(L['EXCLDATE'], format='%Y%m%d', errors='coerce').dt.year
L['fraud_type'] = L['excl_type'].isin(['1128a1', '1128a3', '1128b7'])
L = L.sort_values(['fraud_type', 'excl_year'], ascending=[False, True]).drop_duplicates('NPI')[['NPI', 'excl_year', 'fraud_type']]

def data_year(fn):
    b = os.path.basename(fn)
    m = re.search(r'_DY?(\d{2})_', b, re.I)
    if m: return 2000 + int(m.group(1))
    m = re.search(r'(20[12]\d)', b)
    return int(m.group(1)) if m else None

DROP_PAT = re.compile(r'(last_org_name|first_name|_mi$|_st1|_st2|city|zip|fips|ruca_desc|cntry|crdntls|gndr|race|type_src)', re.I)
CAT_KEYS = {'state_abrvtn': 'State', 'ruca': 'RUCA', 'prscrbr_type': 'Type', 'ent_cd': 'Ent_Cd'}

ONLY = sys.argv[4] if len(sys.argv) > 4 else None   # optional: process one data year only (low-RAM machines)
for f in sorted(glob.glob(os.path.join(src, '*.csv'))):
    Y = data_year(f)
    if Y is None: print('skip (no year in name):', f); continue
    if ONLY and str(Y) != ONLY: continue
    outs = []; cats = None; nrows = 0
    for c in pd.read_csv(f, dtype=str, chunksize=200000):
        nrows += len(c)
        npi = [x for x in c.columns if x.lower() == 'prscrbr_npi'][0]
        c = c.rename(columns={npi: 'NPI'})
        c = c.drop(columns=[x for x in c.columns if x != 'NPI' and DROP_PAT.search(x)])
        if cats is None:
            cats = {}
            for x in list(c.columns):
                lx = x.lower()
                for k, name in CAT_KEYS.items():
                    if lx.endswith(k) and name not in cats: cats[name] = x
        c = c.rename(columns={v: 'cat_' + k for k, v in cats.items()})
        c = c.merge(L, on='NPI', how='left')
        c['years_to_excl'] = c['excl_year'] - Y
        c = c[~(c['years_to_excl'] <= 0)]
        fr = c['fraud_type'].astype('boolean').fillna(False)
        c['label'] = (fr & c['years_to_excl'].between(1, 3)).astype('int8')
        never = c['excl_year'].isna() & c['fraud_type'].isna()
        # same provider-level sampling of negatives as data_prep/03: keep all positives, never-LEIE providers with crc32(NPI) % 4 == 0 (weight 4)
        samp = c['NPI'].map(lambda n: zlib.crc32(n.encode()) % 4 == 0)
        c = c[(c['label'] == 1) | (never & samp)].copy()
        num = [x for x in c.columns if x not in ('NPI', 'excl_year', 'fraud_type', 'years_to_excl', 'label') and not x.startswith('cat_')]
        for x in num: c[x] = pd.to_numeric(c[x], errors='coerce').astype('float32')   # suppressed cells -> NaN
        outs.append(c)
    c = pd.concat(outs, ignore_index=True); del outs
    eps = 1e-6
    g = lambda n: c[n] if n in c.columns else np.nan
    c['cost_per_clm'] = (g('Tot_Drug_Cst') / (g('Tot_Clms') + eps)).astype('float32')
    c['clms_per_bene'] = (g('Tot_Clms') / (g('Tot_Benes') + eps)).astype('float32')
    c['days_per_clm'] = (g('Tot_Day_Suply') / (g('Tot_Clms') + eps)).astype('float32')
    c['brand_share'] = (g('Brnd_Tot_Clms') / (g('Tot_Clms') + eps)).astype('float32')
    c['opioid_share'] = (g('Opioid_Tot_Clms') / (g('Tot_Clms') + eps)).astype('float32')
    c['lis_cost_share'] = (g('LIS_Drug_Cst') / (g('Tot_Drug_Cst') + eps)).astype('float32')
    c.insert(1, 'year', np.int16(Y))
    c = c.drop(columns=['excl_year', 'fraud_type'])
    for k in cats: c['cat_' + k] = c['cat_' + k].astype(str).astype('category')
    c.to_parquet(os.path.join(out, f'partd_year_{Y}.parquet'), index=False)
    print(Y, os.path.basename(f), 'rows_in', nrows, 'kept', len(c), 'pos', int(c['label'].sum()), 'cat', list(cats), flush=True)
