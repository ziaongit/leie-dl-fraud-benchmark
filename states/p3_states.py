"""Paper 3 v2 - independent labels from 32 state Medicaid exclusion lists (OpenSanctions, NPIs and exclusion dates).
Extends the New York analysis to every state list that records NPIs. Positives: provider-years in 2020-2022 that fall one
to three years before a state Medicaid exclusion of a provider never listed in the LEIE ("state-only" providers).
Negatives: all other provider-years in 2020-2022 of providers never in the LEIE and on no state list. Models are tuned and
trained exactly as in the benchmark on LEIE labels for 2013-2019, with all state-listed providers removed from training.
Both classes come from the same 25% provider sample (crc32(NPI) % 4 == 0), so every test row carries the same weight (4, population scale for the audit budgets).
Usage: SDS=phys|partd python p3_states.py   (env: WORK, RMODELS; needs state_medicaid_exclusions_npi_*.csv)"""
import os, sys, glob, time, json
SDS = os.environ.get('SDS', 'phys')
sys.argv = [sys.argv[0], SDS, 'grouped']
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, 'p3_bench.py')] + glob.glob('/kaggle/input/**/p3_bench.py', recursive=True)
src = open([c for c in cands if os.path.exists(c)][0]).read()
src = src[:src.index("done = set()")]
exec(src)
import numpy as np, pandas as pd
WORKS = os.environ.get('WORK', '/kaggle/working/states'); os.makedirs(WORKS, exist_ok=True)
OUTS = f'{WORKS}/p3_states_{SDS}.csv'
MODELS_R = os.environ.get('RMODELS', 'LogReg,LightGBM,XGBoost,MLP,FTT,CNNRNN,AttnCNNRNN').split(',')
done = set(pd.read_csv(OUTS).key) if os.path.exists(OUTS) else set()

def tune_fit(name, tri, vai, seed_base):
    P = Prep(tri); best = (-1, None)
    for ci, cfg in enumerate(configs(name, np.random.default_rng(seed_base))):
        ap, f, _ = fit_score(name, cfg, P, tri, vai, None, np.arange(len(NUM) + len(CAT)), seed=ci)
        if ap > best[0]: best = (ap, f)
    return best[1]

sf = (glob.glob('/kaggle/input/**/state_medicaid_exclusions_npi_*.csv', recursive=True) + glob.glob(os.path.join(os.environ.get('DATA', '.'), '**/state_medicaid_exclusions_npi_*.csv'), recursive=True))[0]
st = pd.read_csv(sf, dtype=str); st = st[st.start_date.notna()]
st['ey'] = st.start_date.str[:4].astype(int)
leie_prov = set(g[y == 1])                                     # providers with a fraud-related LEIE exclusion (others are not in the data)
st = st[~st.NPI.isin(leie_prov)]
first = st.sort_values('ey').drop_duplicates('NPI').set_index('NPI')
on_list = pd.Series(g).isin(set(first.index)).values
ey = pd.Series(g).map(first.ey).values.astype(float)
stt = pd.Series(g).map(first.state).values
pos = on_list & ((ey - yr) >= 1) & ((ey - yr) <= 3)
test = (yr >= 2020) & (yr <= 2022) & (y == 0)
idx = np.where(test & (~on_list | pos))[0]
yt = pos[idx].astype(int)
print(SDS, 'state-only providers', len(first), '| test rows', len(idx), 'positives', int(yt.sum()), 'providers', len(set(g[idx][yt == 1])), flush=True)
print('positives by state:', pd.Series(stt[idx][yt == 1]).value_counts().head(12).to_dict(), flush=True)
tr = np.where((yr <= 2019) & ~on_list)[0]
rng = np.random.default_rng(7)
vh = np.array([zlib.crc32((n + 'v0').encode()) % 100 for n in g[tr]]); trn, va = tr[vh >= 15], tr[vh < 15]
neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
big = [s for s, n in pd.Series(stt[idx][yt == 1]).value_counts().items() if n >= 40]
for name in MODELS_R:
    k = f'states_{SDS}_{name}'
    if k in done: continue
    t0 = time.time(); f = tune_fit(name, tri, vai, 900); s = f(idx)
    r = dict(key=k, dataset=SDS, model=name, scope='all states', n_test=len(idx), positives=int(yt.sum()), secs=round(time.time() - t0))
    r.update(metrics(yt, s, np.full(len(idx), 4.0))); rows = [r]
    for stname in big:                                         # per state: that state's positives against all negatives
        m = (yt == 0) | (stt[idx] == stname)
        rr = dict(key=f'{k}_{stname}', dataset=SDS, model=name, scope=stname, n_test=int(m.sum()), positives=int(yt[m].sum()), secs=0)
        rr.update(metrics(yt[m], s[m], np.full(int(m.sum()), 4.0))); rows.append(rr)
    pd.DataFrame(rows).to_csv(OUTS, mode='a', header=not os.path.exists(OUTS), index=False)
    np.save(f'{WORKS}/states_{SDS}_{name}.npy', s.astype(np.float32))
    print('states', SDS, name, 'ROC', round(r['roc_auc'], 3), 'lift', round(r['lift_pr'], 2), 'P@1000', round(r['prec_at_1000'], 4), 'pos', r['positives'], r['secs'], 's', flush=True)
print('STATES DONE', SDS, flush=True)
