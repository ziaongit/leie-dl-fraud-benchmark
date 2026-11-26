"""Build the table of state Medicaid exclusions with NPIs from the OpenSanctions FollowTheMoney exports
(one file per state dataset, e.g. us_ca_med_exclusions, us_il_med_exclusions, ...; release of 16 November 2026 used in the paper).
Usage: python data_prep/06_state_exclusions.py <folder with us_<st>_med_exclusions*.json (FtM lines)> <out_csv>
Output columns: NPI, state, start_date (earliest sanction start date of that NPI on that state's list; empty if the list gives none).
Only lists that record NPI numbers contribute (32 states in the release used)."""
import json, sys, glob, re, os, pandas as pd
src, out = sys.argv[1], sys.argv[2]; rows = []
for f in sorted(glob.glob(os.path.join(src, 'us_*_med_exclusions*.json'))):
    st = re.search(r'us_([a-z]{2})_med_exclusions', os.path.basename(f)).group(1).upper(); ent, start = {}, {}
    for line in open(f, encoding='utf-8'):
        e = json.loads(line); p = e.get('properties', {})
        if e['schema'] in ('LegalEntity', 'Person', 'Company', 'Organization') and p.get('npiCode'):
            ent[e['id']] = [n.strip() for n in p['npiCode']]
        elif e['schema'] == 'Sanction':
            for t in p.get('entity', []):
                for s in p.get('startDate', []): start[t] = min(start.get(t, s), s)
    for i, npis in ent.items():
        for n in npis:
            if re.fullmatch(r'[1-9]\d{9}', n): rows.append(dict(NPI=n, state=st, start_date=(start.get(i) or '')[:10]))
d = pd.DataFrame(rows).sort_values(['NPI', 'state', 'start_date']).drop_duplicates(['NPI', 'state'])
d.to_csv(out, index=False); print(len(d), 'rows,', d.state.nunique(), 'states')
