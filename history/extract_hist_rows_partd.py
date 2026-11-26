"""Part D history rows: earlier years (more than three years before exclusion) of providers with at least one positive
Part D provider-year, built with exactly the same steps as data_prep/04_build_partd.py. Input: raw CMS CSVs pre-filtered to
those providers' NPIs (header + matching lines). The fraud-window rows (1-3 years before exclusion) are rebuilt too and
compared with the benchmark sample as a check; only the earlier rows are written. Labels of these rows are never used."""
import pandas as pd, numpy as np, re, glob, os, sys
rawdir, leie_path, samp_dir, out = sys.argv[1:5]
L = pd.read_csv(leie_path, dtype=str, keep_default_na=False)
L['NPI'] = L['NPI'].str.strip(); L = L[L['NPI'].str.fullmatch(r'[1-9]\d{9}')].copy()
L['excl_type'] = L['EXCLTYPE'].str.strip().str.lower()
L['excl_year'] = pd.to_datetime(L['EXCLDATE'], format='%Y%m%d', errors='coerce').dt.year
L['fraud_type'] = L['excl_type'].isin(['1128a1', '1128a3', '1128b7'])
L = L.sort_values(['fraud_type', 'excl_year'], ascending=[False, True]).drop_duplicates('NPI')[['NPI', 'excl_year', 'fraud_type']]
DROP_PAT = re.compile(r'(last_org_name|first_name|_mi$|_st1|_st2|city|zip|fips|ruca_desc|cntry|crdntls|gndr|race|type_src)', re.I)
CAT_KEYS = {'state_abrvtn': 'State', 'ruca': 'RUCA', 'prscrbr_type': 'Type', 'ent_cd': 'Ent_Cd'}
cols = pd.read_parquet(sorted(glob.glob(f'{samp_dir}/partd_year_*.parquet'))[0]).columns.tolist()
parts = []
for f in sorted(glob.glob(f'{rawdir}/dy*.csv')):
    Y = 2000 + int(re.search(r'dy(\d\d)', f).group(1))
    c = pd.read_csv(f, dtype=str)
    npi = [x for x in c.columns if x.lower() == 'prscrbr_npi'][0]; c = c.rename(columns={npi: 'NPI'})
    c = c[c.NPI.str.fullmatch(r'[1-9]\d{9}', na=False)]
    c = c.drop(columns=[x for x in c.columns if x != 'NPI' and DROP_PAT.search(x)])
    cats = {}
    for x in list(c.columns):
        for k, name in CAT_KEYS.items():
            if x.lower().endswith(k) and name not in cats: cats[name] = x
    c = c.rename(columns={v: 'cat_' + k for k, v in cats.items()})
    c = c.merge(L, on='NPI', how='inner'); c = c[c.fraud_type.astype(bool)]
    c['years_to_excl'] = c['excl_year'] - Y
    c = c[c.years_to_excl > 0].copy()
    c['label'] = c.years_to_excl.between(1, 3).astype('int8')
    num = [x for x in c.columns if x not in ('NPI', 'excl_year', 'fraud_type', 'years_to_excl', 'label') and not x.startswith('cat_')]
    for x in num: c[x] = pd.to_numeric(c[x], errors='coerce').astype('float32')
    eps = 1e-6; g = lambda n: c[n] if n in c.columns else np.nan
    c['cost_per_clm'] = (g('Tot_Drug_Cst') / (g('Tot_Clms') + eps)).astype('float32'); c['clms_per_bene'] = (g('Tot_Clms') / (g('Tot_Benes') + eps)).astype('float32')
    c['days_per_clm'] = (g('Tot_Day_Suply') / (g('Tot_Clms') + eps)).astype('float32'); c['brand_share'] = (g('Brnd_Tot_Clms') / (g('Tot_Clms') + eps)).astype('float32')
    c['opioid_share'] = (g('Opioid_Tot_Clms') / (g('Tot_Clms') + eps)).astype('float32'); c['lis_cost_share'] = (g('LIS_Drug_Cst') / (g('Tot_Drug_Cst') + eps)).astype('float32')
    c['year'] = np.int16(Y)
    for k in cats: c['cat_' + k] = c['cat_' + k].astype(str)
    s = pd.read_parquet(f'{samp_dir}/partd_year_{Y}.parquet', columns=['NPI', 'label', 'Tot_Clms']); sp = s[s.label == 1]
    w = c[c.label == 1].merge(sp, on='NPI', suffixes=('', '_s'))
    print(Y, 'earlier rows', int((c.label == 0).sum()), '| window rows rebuilt', int((c.label == 1).sum()), 'sample positives', len(sp), 'matched', len(w), 'Tot_Clms equal', bool(np.allclose(w.Tot_Clms, w.Tot_Clms_s, equal_nan=True)), flush=True)
    e = c[c.label == 0].copy(); e['label'] = -1
    missing = [x for x in cols if x not in e.columns]
    for x in missing: e[x] = np.nan
    parts.append(e[cols])
x = pd.concat(parts, ignore_index=True); x.to_parquet(out, index=False); print('rows', len(x), 'providers', x.NPI.nunique(), 'cols', len(cols))
