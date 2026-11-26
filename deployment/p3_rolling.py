"""Paper 3 v2 - retrospective deployment simulation (rolling-origin backtest).
For each deployment year s = 2016..2022 a payer retrains on what it could know at the end of data year s and scores the
providers billing in year s:
  * training rows: data years 2013..s-1 of the study sample; a row is positive only if its fraud-related LEIE exclusion
    (year + years_to_excl) had happened by the end of year s, otherwise it counts as negative (fraud not yet known);
  * test rows: all provider-years of data year s in the sample (providers already excluded are not in the data);
    outcome = fraud-related exclusion one to three years later (the study label);
  * the top 1,000 and 5,000 provider-years (population scale, negatives weight 4) are 'audited'.
Same tuning protocol (6 random configurations, inner provider-grouped validation, class weights) as the benchmark.
Lead time: for every fraud provider-year flagged, years until exclusion; for every fraud provider, the earliest deployment
year in which it was flagged versus its exclusion year. Assumes data are available at the end of each year; the CMS
publication lag (about 1.5-2 years) would shorten real lead times accordingly.
Usage: SDS=phys|partd python p3_rolling.py   (env: WORK, RMODELS)"""
import os, sys, glob, time, json, zlib
SDS = os.environ.get('SDS', 'phys')
sys.argv = [sys.argv[0], SDS, 'grouped']
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, 'p3_bench.py')] + glob.glob('/kaggle/input/**/p3_bench.py', recursive=True)
src = open([c for c in cands if os.path.exists(c)][0]).read()
src = src.replace("X = d.drop(columns=[ID, 'year', 'label', 'years_to_excl']); del d",
                  "YTX = d['years_to_excl'].values.astype(float); X = d.drop(columns=[ID, 'year', 'label', 'years_to_excl']); del d")
assert 'YTX =' in src
src = src[:src.index("done = set()")]
exec(src)
import numpy as np, pandas as pd
WORKR = os.environ.get('WORK', '/kaggle/working/rolling'); os.makedirs(WORKR, exist_ok=True)
OUTR = f'{WORKR}/p3_rolling_{SDS}.csv'; OUTF = f'{WORKR}/p3_rolling_flags_{SDS}.csv'
MODELS_R = os.environ.get('RMODELS', 'LightGBM,XGBoost,MLP,LogReg').split(',')
excl_year = np.where(y == 1, yr + YTX, np.nan)                       # exclusion year of positive rows
done = set(pd.read_csv(OUTR).key) if os.path.exists(OUTR) else set()

def tune_fit(name, tri, vai, ytr, seed_base):
    global y
    y_save = y.copy(); y[:] = ytr                                    # labels as known at deployment time
    try:
        P = Prep(tri); best = (-1, None)
        for ci, cfg in enumerate(configs(name, np.random.default_rng(seed_base))):
            ap, f, _ = fit_score(name, cfg, P, tri, vai, None, np.arange(len(NUM) + len(CAT)), seed=ci)
            if ap > best[0]: best = (ap, f)
    finally:
        y[:] = y_save
    return best[1]

for s_year in [int(v) for v in os.environ.get('YEARS', '2016,2017,2018,2019,2020,2021,2022').split(',')]:
    known = (y == 1) & (excl_year <= s_year)                           # fraud known by the end of year s
    ytr = known.astype(np.int8)
    tr = np.where(yr <= s_year - 1)[0]; te = np.where(yr == s_year)[0]
    rng = np.random.default_rng(s_year)
    vh = np.array([zlib.crc32((n + 'roll').encode()) % 100 for n in g[tr]]); trn, va = tr[vh >= 15], tr[vh < 15]
    neg = trn[ytr[trn] == 0]; tri = np.concatenate([trn[ytr[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
    vneg = va[ytr[va] == 0]; vai = np.concatenate([va[ytr[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
    for name in MODELS_R:
        k = f'roll_{SDS}_{s_year}_{name}'
        if k in done: continue
        t0 = time.time(); f = tune_fit(name, tri, vai, ytr, 700 + s_year); sc = f(te)
        r = dict(key=k, dataset=SDS, year=s_year, model=name, n_train_pos=int(ytr[tri].sum()), n_test=len(te), pos_test=int(y[te].sum()),
                 pos_providers=len(set(g[te][y[te] == 1])), secs=round(time.time() - t0))
        r.update(metrics(y[te], sc, w_all[te]))
        o = np.argsort(-sc); cw = np.cumsum(w_all[te][o]); fl = []
        for kk in (1000, 5000):
            sel = o[:np.searchsorted(cw, kk) + 1]; tp = sel[y[te][sel] == 1]
            r[f'cases_at_{kk}'] = int(len(tp)); r[f'audits_per_case_{kk}'] = float(w_all[te][sel].sum() / max(len(tp), 1))
            r[f'lead_mean_{kk}'] = float(YTX[te][tp].mean()) if len(tp) else np.nan
            for i in tp: fl.append(dict(dataset=SDS, model=name, year=s_year, k=kk, npi=g[te][i], excl_year=float(excl_year[te][i]), ytx=float(YTX[te][i])))
        pd.DataFrame([r]).to_csv(OUTR, mode='a', header=not os.path.exists(OUTR), index=False)
        if fl: pd.DataFrame(fl).to_csv(OUTF, mode='a', header=not os.path.exists(OUTF), index=False)
        print('roll', SDS, s_year, name, 'pos', r['pos_test'], 'cases@1000', r['cases_at_1000'], 'cases@5000', r['cases_at_5000'],
              'lift', round(r['lift_pr'], 1), 'ROC', round(r['roc_auc'], 3), 'lead', round(r['lead_mean_1000'], 2) if r['cases_at_1000'] else '-', r['secs'], 's', flush=True)
print('ROLLING DONE', SDS, flush=True)
