"""Rows needed for provider histories that are not in the benchmark sample: the earlier years (more than three years
before exclusion) of providers with a fraud-related exclusion that have at least one positive provider-year.
Features are derived exactly as in data_prep/02_assemble_physician.py. Labels of these rows are not used (history only)."""
import pandas as pd, numpy as np, glob, os, sys
full, samp, out = sys.argv[1], sys.argv[2], sys.argv[3]
cols = pd.read_parquet(sorted(glob.glob(f'{samp}/provider_year_*.parquet'))[0]).columns.tolist()
pos = set(pd.concat([pd.read_parquet(f, columns=['Rndrng_NPI', 'label']) for f in sorted(glob.glob(f'{samp}/provider_year_*.parquet'))]).query('label == 1').Rndrng_NPI)
parts = []
for f in sorted(glob.glob(f'{full}/provider_year_*.parquet')):
    cache = f'{os.path.dirname(out)}/_parts/{os.path.basename(f)}'
    if os.path.exists(cache): parts.append(pd.read_parquet(cache)); print('cached', cache, flush=True); continue
    import pyarrow.parquet as pq
    have = pq.read_schema(f).names
    d = pd.read_parquet(f, columns=[c for c in have if c in cols or c in ('years_to_excl',)]); d = d[d.Rndrng_NPI.isin(pos) & (d.years_to_excl > 3)].copy()
    num = [c for c in cols if c not in ('Rndrng_NPI', 'year', 'label', 'years_to_excl', 'Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'srv_per_bene', 'pay_per_bene', 'pay_per_srv', 'chrg_to_allowed', 'hcpcs_per_bene', 'drug_pay_share')]
    for c in num: d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
    eps = 1e-6; a = d
    a['srv_per_bene'] = (a['Tot_Srvcs'] / (a['Tot_Benes'] + eps)).astype('float32'); a['pay_per_bene'] = (a['Tot_Mdcr_Pymt_Amt'] / (a['Tot_Benes'] + eps)).astype('float32')
    a['pay_per_srv'] = (a['Tot_Mdcr_Pymt_Amt'] / (a['Tot_Srvcs'] + eps)).astype('float32'); a['chrg_to_allowed'] = (a['Tot_Sbmtd_Chrg'] / (a['Tot_Mdcr_Alowd_Amt'] + eps)).astype('float32')
    a['hcpcs_per_bene'] = (a['Tot_HCPCS_Cds'] / (a['Tot_Benes'] + eps)).astype('float32'); a['drug_pay_share'] = (a['Drug_Mdcr_Pymt_Amt'] / (a['Tot_Mdcr_Pymt_Amt'] + eps)).astype('float32')
    a['label'] = -1; a['year'] = a['year'].astype('int16')
    for c in ('Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind'): a[c] = a[c].astype(str)
    part = a[[c for c in cols if c in a.columns]]; os.makedirs(os.path.dirname(cache), exist_ok=True); part.to_parquet(cache, index=False)
    parts.append(part); print(os.path.basename(f), len(a), flush=True)
x = pd.concat(parts, ignore_index=True); x.to_parquet(out, index=False); print('rows', len(x), 'providers', x.Rndrng_NPI.nunique(), 'missing cols', set(cols) - set(x.columns))
