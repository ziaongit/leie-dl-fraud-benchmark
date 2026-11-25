"""Build the New York State Medicaid (OMIG) exclusion table with NPIs from the OpenSanctions FollowTheMoney export.
Usage: python data_prep/05_ny_exclusions.py <us_ny_med_exclusions .json (FtM lines)> <LEIE csv> <out_csv> [provider_year_dir]
Output columns: NPI, sector, start (earliest sanction start date), year, in_leie, in_cms."""
import json, sys, glob, pandas as pd
src, leie, out = sys.argv[1], sys.argv[2], sys.argv[3]; pdir = sys.argv[4] if len(sys.argv) > 4 else None
ent, start = {}, {}
for line in open(src, encoding='utf-8'):
    e = json.loads(line); p = e.get('properties', {})
    if e['schema'] in ('LegalEntity', 'Person', 'Company', 'Organization') and p.get('npiCode'):
        ent[e['id']] = (p['npiCode'][0].strip(), (p.get('sector') or [''])[0])
    elif e['schema'] == 'Sanction':
        for t in p.get('entity', []):
            for s in p.get('startDate', []): start[t] = min(start.get(t, s), s)
rows = [dict(NPI=n, sector=sec, start=start.get(i)) for i, (n, sec) in ent.items()]
d = pd.DataFrame(rows).dropna(subset=['start'])
d = d[d.NPI.str.fullmatch(r'[1-9]\d{9}')].sort_values('start').drop_duplicates('NPI')
d['year'] = d.start.str[:4].astype(int)
L = pd.read_csv(leie, dtype=str, keep_default_na=False); d['in_leie'] = d.NPI.isin(set(L.NPI.str.strip()))
if pdir:
    cms = set()
    for f in glob.glob(f'{pdir}/provider_year_*.parquet'): cms |= set(pd.read_parquet(f, columns=['Rndrng_NPI']).Rndrng_NPI)
    d['in_cms'] = d.NPI.isin(cms)
d.to_csv(out, index=False); print(len(d), 'NPIs;', int(d.in_leie.sum()), 'also in LEIE')
