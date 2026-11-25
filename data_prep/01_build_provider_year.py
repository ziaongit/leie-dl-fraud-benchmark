"""Step 1: build one provider-year file with LEIE labels.
Usage: python data_prep/01_build_provider_year.py <cms_by_provider_csv> <data_year> <leie_csv> <out_dir>
Label (main): fraud-related exclusion (1128a1, 1128a3, 1128b7) with valid NPI, excluded 1-3 years after the data year.
Provider-years in or after the exclusion year are dropped. Other LEIE types are kept and flagged for sensitivity analyses."""
import pandas as pd, sys, os
cms_path, Y, leie_path, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
L = pd.read_csv(leie_path, dtype=str, keep_default_na=False)
L['NPI'] = L['NPI'].str.strip(); L = L[L['NPI'].str.fullmatch(r'[1-9]\d{9}')].copy()
L['excl_type'] = L['EXCLTYPE'].str.strip().str.lower()
L['excl_year'] = pd.to_datetime(L['EXCLDATE'], format='%Y%m%d', errors='coerce').dt.year
L['fraud_type'] = L['excl_type'].isin(['1128a1', '1128a3', '1128b7'])
L = L.sort_values(['fraud_type', 'excl_year'], ascending=[False, True]).drop_duplicates('NPI')[['NPI', 'excl_type', 'excl_year', 'fraud_type']]
c = pd.read_csv(cms_path, dtype={'Rndrng_NPI': str, 'Rndrng_Prvdr_RUCA': str, 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind': str, 'Drug_Sprsn_Ind': str, 'Med_Sprsn_Ind': str}, low_memory=False)
drop = ['Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name', 'Rndrng_Prvdr_MI', 'Rndrng_Prvdr_St1', 'Rndrng_Prvdr_St2', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_RUCA_Desc', 'Rndrng_Prvdr_Cntry', 'Rndrng_Prvdr_State_FIPS']
c = c.drop(columns=[x for x in drop if x in c.columns])
c.insert(1, 'year', Y)
c = c.merge(L, left_on='Rndrng_NPI', right_on='NPI', how='left').drop(columns='NPI')
c['years_to_excl'] = c['excl_year'] - Y
n0 = len(c)
c = c[~(c['years_to_excl'] <= 0)]  # drop provider-years in or after the exclusion year
c['label'] = (c['fraud_type'].astype('boolean').fillna(False) & c['years_to_excl'].between(1, 3)).astype(int)
os.makedirs(out, exist_ok=True)
c.to_parquet(os.path.join(out, f'provider_year_{Y}.parquet'), index=False)
print(Y, 'rows', n0, 'kept', len(c), 'dropped_post_excl', n0 - len(c), 'pos', int(c['label'].sum()),
      'fraud_any_future', int((c['fraud_type'].astype('boolean').fillna(False) & (c['years_to_excl'] > 0)).sum()),
      'other_leie', int(((~c['fraud_type'].astype('boolean').fillna(True)) & (c['years_to_excl'] > 0)).sum()))
