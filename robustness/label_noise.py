"""Label-noise robustness: LEIE captures only detected fraud, so some 'negatives' are undetected fraud.
Simulate this by hiding a share q of the TRAINING positives (relabel as 0) and check whether
(i) test performance against the full LEIE labels and (ii) the ranking of models change.
Provider-grouped 5-fold CV on Physician data; models: LightGBM, XGBoost, logistic regression, MLP (sklearn).
Hiding is done at provider level (all positive years of a hidden provider), q in {0, 0.1, 0.25, 0.5}."""
import pandas as pd, numpy as np, glob, zlib, time, os
from pandas.api.types import union_categoricals
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb, xgboost as xgb
import sys
PHYS = sys.argv[1] if len(sys.argv) > 1 else 'data/phys'   # sampled Physician parquets (data_prep/03)
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/label_noise_results.csv'
CAT = ['Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind']
parts = [pd.read_parquet(f) for f in sorted(glob.glob(f'{PHYS}/provider_year_*.parquet'))]
for c in CAT:
    u = union_categoricals([p[c].astype(str).astype('category') for p in parts]).categories
    for p in parts: p[c] = pd.Categorical(p[c].astype(str), categories=u)
d = pd.concat(parts, ignore_index=True); del parts
y = d.label.values.astype(int); g = d.Rndrng_NPI.values; w = np.where(y == 1, 1.0, 4.0)
X = d.drop(columns=['Rndrng_NPI', 'year', 'label', 'years_to_excl']); del d
NUM = [c for c in X.columns if c not in CAT]
XN = np.log1p(np.clip(np.nan_to_num(X[NUM].values.astype(np.float32)), 0, None))
TY = X['Rndrng_Prvdr_Type'].cat.codes.values
done = set()
if os.path.exists(OUT): p = pd.read_csv(OUT); done = set(zip(p.q, p.fold, p.model))
salt = np.array([zlib.crc32((n + '0').encode()) for n in g]) % 5
pos_prov = np.array(sorted(set(g[y == 1])))
for fold in range(5):
    te = np.where(salt == fold)[0]; tr0 = np.where(salt != fold)[0]
    rng = np.random.default_rng(fold); neg = tr0[y[tr0] == 0]
    tr = np.concatenate([tr0[y[tr0] == 1], rng.choice(neg, size=len(neg) // 10, replace=False)])
    trprov = np.array(sorted(set(g[tr][y[tr] == 1])))
    for q in [0.0, 0.1, 0.25, 0.5]:
        hide = set(np.random.default_rng(100 + fold).choice(trprov, size=int(len(trprov) * q), replace=False))
        yn = y[tr].copy(); yn[np.isin(g[tr], list(hide))] = 0
        pw = 4.0 * (yn == 0).sum() / max(1, yn.sum())
        Z = lambda idx: np.hstack([XN[idx], TY[idx, None]])
        mu, sd = XN[tr].mean(0), XN[tr].std(0) + 1e-6
        S = lambda idx: (XN[idx] - mu) / sd
        for name in ['LightGBM', 'XGBoost', 'LogReg', 'MLP']:
            if (q, fold, name) in done: continue
            t0 = time.time()
            if name == 'LightGBM':
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, reg_lambda=5, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.7, scale_pos_weight=pw, n_jobs=2, verbose=-1, random_state=0).fit(X.iloc[tr], yn, categorical_feature=CAT)
                s = m.predict_proba(X.iloc[te])[:, 1]
            elif name == 'XGBoost':
                m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                                      scale_pos_weight=pw, tree_method='hist', n_jobs=2, random_state=0).fit(Z(tr), yn)
                s = m.predict_proba(Z(te))[:, 1]
            elif name == 'LogReg':
                m = LogisticRegression(C=0.1, class_weight={0: 1, 1: pw}, max_iter=400).fit(S(tr), yn); s = m.predict_proba(S(te))[:, 1]
            else:
                pos = np.where(yn == 1)[0]; idx = np.concatenate([np.arange(len(yn))] + [pos] * 20)
                m = MLPClassifier((128, 64), early_stopping=True, max_iter=40, random_state=0).fit(S(tr)[idx], yn[idx]); s = m.predict_proba(S(te))[:, 1]
            yt, wt = y[te], w[te]; prev = (yt * wt).sum() / wt.sum(); o = np.argsort(-s); cut = np.searchsorted(np.cumsum(wt[o]), 1000)
            r = dict(q=q, fold=fold, model=name, hidden_providers=len(hide), pr_auc=average_precision_score(yt, s, sample_weight=wt),
                     roc_auc=roc_auc_score(yt, s, sample_weight=wt), p_at_1000=(yt[o[:cut + 1]] * wt[o[:cut + 1]]).sum() / wt[o[:cut + 1]].sum(), secs=round(time.time() - t0))
            r['lift_pr'] = r['pr_auc'] / prev
            pd.DataFrame([r]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            print(q, fold, name, round(r['pr_auc'], 5), round(r['roc_auc'], 3), r['secs'], flush=True)
print('ALL DONE')
