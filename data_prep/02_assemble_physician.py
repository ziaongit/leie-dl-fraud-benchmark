"""Step 2: assemble the analysis dataset (all years) with features.
Main label definition:
  label = 1 : provider-year 1-3 years before a fraud-related LEIE exclusion (1128a1/a3/b7)
  label = 0 : provider never appears in the LEIE (any type, any date)
  dropped   : provider-years of providers with other LEIE exclusions, or fraud exclusions > 3 years later
Features: provider-year utilisation, payment, beneficiary-mix and specialty variables only;
nothing derived from LEIE or from the label."""
import pandas as pd, numpy as np, glob, sys, os
src = sys.argv[1] if len(sys.argv) > 1 else 'data/provider_year'
out = sys.argv[2] if len(sys.argv) > 2 else 'data/analysis'
files = sorted(glob.glob(os.path.join(src, 'provider_year_*.parquet')))
cols = [set(pd.read_parquet(f, columns=None).columns) if False else set(__import__('pyarrow.parquet', fromlist=['x']).read_schema(f).names) for f in files]
common = set.intersection(*cols)
ID = ['Rndrng_NPI', 'year']
CAT = ['Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind']
LAB = ['label', 'excl_type', 'excl_year', 'years_to_excl', 'fraud_type']
NUM = sorted(c for c in common if c not in ID + CAT + LAB + ['Rndrng_Prvdr_Crdntls', 'Drug_Sprsn_Ind', 'Med_Sprsn_Ind'])
for f in files:
    d = pd.read_parquet(f, columns=ID + CAT + LAB + NUM)
    never = d['excl_type'].isna()
    keep = (d['label'] == 1) | never
    d = d[keep].copy()
    for c in NUM: d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
    for c in CAT: d[c] = d[c].astype(str).astype('category')
    d['label'] = d['label'].astype('int8'); d['year'] = d['year'].astype('int16')
    d = d.drop(columns=['excl_type', 'excl_year', 'fraud_type'])
    a = d; eps = 1e-6
    a['srv_per_bene'] = (a['Tot_Srvcs'] / (a['Tot_Benes'] + eps)).astype('float32')
    a['pay_per_bene'] = (a['Tot_Mdcr_Pymt_Amt'] / (a['Tot_Benes'] + eps)).astype('float32')
    a['pay_per_srv'] = (a['Tot_Mdcr_Pymt_Amt'] / (a['Tot_Srvcs'] + eps)).astype('float32')
    a['chrg_to_allowed'] = (a['Tot_Sbmtd_Chrg'] / (a['Tot_Mdcr_Alowd_Amt'] + eps)).astype('float32')
    a['hcpcs_per_bene'] = (a['Tot_HCPCS_Cds'] / (a['Tot_Benes'] + eps)).astype('float32')
    a['drug_pay_share'] = (a['Drug_Mdcr_Pymt_Amt'] / (a['Tot_Mdcr_Pymt_Amt'] + eps)).astype('float32')
    os.makedirs(out, exist_ok=True)
    a.to_parquet(os.path.join(out, os.path.basename(f)), index=False)
    print(os.path.basename(f), len(a), int(a['label'].sum()), flush=True)
    del a, d
print('numeric features', len(NUM) + 6, 'categorical', len(CAT))
print('NUM:', NUM)
