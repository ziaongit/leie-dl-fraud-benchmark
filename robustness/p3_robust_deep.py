"""Paper 3 v2 - robustness analyses for ALL benchmark models (answers mock-review M3 / IJDSA R2-15).
Uses p3_bench.py (data loading, preprocessing, models, tuning) unchanged.

Part A - label noise (Physician, grouped repeat 0, 5 folds): a share q of the positive providers in the TRAINING folds is
         relabelled negative (also in the inner validation set, as in practice), every model is tuned and trained as in the
         benchmark, and test folds are scored against the full LEIE labels. q in {0, 0.25, 0.5}.
Part B - independent labels (Physician, New York): models are trained on LEIE labels for data years 2013-2019 with providers
         on the New York Medicaid exclusion list removed, and tested on ALL New York provider-years 2020-2022:
           (i)  positives = provider-years 1-3 years before a New York exclusion of a provider never in the LEIE;
           (ii) positives = LEIE fraud positives in New York (comparison).
         Negatives = New York providers never in the LEIE and not on the New York-only list; training excludes New York-only providers. The 25% sample used in the benchmark
         is completed with the remaining New York rows (ny_extra_rows_<year>.parquet), so the New York test set is the full population (weight 1).
Usage on Kaggle: python p3_robust_deep.py noise   |   python p3_robust_deep.py ny"""
import os, sys, glob, zlib, json, time, numpy as np, pandas as pd
PART = sys.argv[1] if len(sys.argv) > 1 else 'noise'
sys.argv = [sys.argv[0], 'phys', 'grouped']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p3_bench.py')).read()
src = src.replace("NCAT = [len(X[c].cat.categories) for c in CAT]; del X", "NCAT = [len(X[c].cat.categories) for c in CAT]; CATS = {c: list(X[c].cat.categories) for c in CAT}; del X")
src = src[:src.index("done = set()")]
exec(src)
MODELS_R = os.environ.get('RMODELS', 'LogReg,LightGBM,XGBoost,MLP,FTT,CNNRNN,AttnCNNRNN').split(',')
OUTR = f'{WORK}/p3_robust_{PART}.csv'
done = set()
if os.path.exists(OUTR): done = set(pd.read_csv(OUTR).key)

def tune_fit(name, tri, vai, ylab, seed_base):
    """tune with the benchmark budget on (tri, vai) using labels ylab; returns a predict function"""
    global y
    y_true = y; y = ylab                                   # fit_score / train_nn read the global y
    try:
        P = Prep(tri); best = (-1, None)
        for ci, cfg in enumerate(configs(name, np.random.default_rng(seed_base))):
            ap, f, _ = fit_score(name, cfg, P, tri, vai, None, np.arange(len(NUM) + len(CAT)), seed=ci)
            if ap > best[0]: best = (ap, f)
    finally:
        y = y_true
    return best[1]

def write(r):
    pd.DataFrame([r]).to_csv(OUTR, mode='a', header=not os.path.exists(OUTR), index=False); print(r, flush=True)

if PART == 'noise':
    rep = 0; salt = np.array([zlib.crc32((n + str(rep)).encode()) for n in g]) % FOLDS
    for fold in range(int(os.environ.get('NOISE_FOLDS', FOLDS))):
        tr, te = np.where(salt != fold)[0], np.where(salt == fold)[0]
        rng = np.random.default_rng(1000 * rep + fold)
        vh = np.array([zlib.crc32((n + 'v' + str(rep)).encode()) % 100 for n in g[tr]])
        trn, va = tr[vh >= 15], tr[vh < 15]
        neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
        vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
        posprov = np.array(sorted(set(g[tr][y[tr] == 1])))
        for q in [0.0, 0.25, 0.5]:
            hide = set(np.random.default_rng(100 + fold).choice(posprov, size=int(len(posprov) * q), replace=False))
            intr = np.zeros(len(y), bool); intr[tr] = True
            yn = y.copy(); yn[np.isin(g, list(hide)) & intr] = 0
            for name in MODELS_R:
                key = f'noise_{fold}_{q}_{name}'
                if key in done: continue
                t0 = time.time(); f = tune_fit(name, tri, vai, yn, 500 + 10 * fold)
                r = dict(key=key, part='noise', fold=fold, q=q, model=name, hidden_providers=len(hide), secs=round(time.time() - t0))
                r.update(metrics(y[te], f(te), w_all[te])); write(r)
else:
    ny = pd.read_csv(glob.glob('/kaggle/input/**/ny_npi_exclusions.csv', recursive=True)[0] if os.path.exists('/kaggle/input') else 'data/ny_npi_exclusions.csv', dtype={'NPI': str})
    ny_only = ny[~ny.in_leie].groupby('NPI').year.min(); ny_all = set(ny.NPI)
    # append the New York rows that are not in the 25% sample
    x = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{DATA}/**/ny_extra_rows_*.parquet', recursive=True))], ignore_index=True)
    xc = np.stack([pd.Categorical(x[c].astype(str), categories=CATS[c]).codes for c in CAT], 1).astype(np.int32)
    n0 = len(y)
    XN = np.vstack([XN, x[NUM].values.astype(np.float32)]); XC = np.vstack([XC, xc])
    y = np.concatenate([y, x.label.values.astype(np.int8)]); g = np.concatenate([g, x.Rndrng_NPI.values]); yr = np.concatenate([yr, x.year.values.astype(int)])
    w_all = np.concatenate([w_all, np.ones(len(x), np.float32)])
    state_ny = np.concatenate([XC[:n0, CAT.index('Rndrng_Prvdr_State_Abrvtn')] == CATS['Rndrng_Prvdr_State_Abrvtn'].index('NY'), np.ones(len(x), bool)])
    on_ny = pd.Series(g).isin(set(ny_only.index)).values      # providers on the New York list but never in the LEIE (as in ny_external.py)
    ny_year = pd.Series(g).map(ny_only).values.astype(float)
    ny_pos = ((ny_year - yr) >= 1) & ((ny_year - yr) <= 3)
    # training: 2013-2019, sampled rows only, providers on the New York list removed
    tr = np.where((np.arange(len(y)) < n0) & (yr <= 2019) & ~on_ny)[0]
    rng = np.random.default_rng(7)
    vh = np.array([zlib.crc32((n + 'v0').encode()) % 100 for n in g[tr]]); trn, va = tr[vh >= 15], tr[vh < 15]
    neg = trn[y[trn] == 0]; tri = np.concatenate([trn[y[trn] == 1], rng.choice(neg, size=int(len(neg) * NEG_FRAC), replace=False)])
    vneg = va[y[va] == 0]; vai = np.concatenate([va[y[va] == 1], rng.choice(vneg, size=min(len(vneg), 30000), replace=False)])
    test = state_ny & (yr >= 2020) & (yr <= 2022)
    clean_neg = test & (y == 0) & ~on_ny
    sets = {'ny_only_exclusions': (np.where(clean_neg | (test & ny_pos & (y == 0)))[0], 'ny'),
            'leie_in_ny': (np.where(clean_neg | (test & (y == 1) & ~on_ny))[0], 'leie')}
    for name in MODELS_R:
        if all(f'ny_{s}_{name}' in done for s in sets): continue
        t0 = time.time(); f = tune_fit(name, tri, vai, y, 900)
        for sname, (idx, kind) in sets.items():
            yt = (ny_pos[idx] & (y[idx] == 0)).astype(int) if kind == 'ny' else y[idx].astype(int)
            wt = np.ones(len(idx)); s = f(idx); r = metrics(yt, s, wt)
            o = np.argsort(-s); top = o[:max(1, len(o) // 100)]
            r.update(dict(key=f'ny_{sname}_{name}', part='ny', label_source=sname, model=name, n_test=len(idx), positives=int(yt.sum()),
                          lift_top1pct=yt[top].mean() / yt.mean(), secs=round(time.time() - t0)))
            write(r)
print('ROBUST DONE', PART, flush=True)
