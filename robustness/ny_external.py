"""External label check: do models trained on LEIE fraud labels also rank providers excluded by the
New York State Medicaid programme (OMIG list, not in LEIE) above other New York providers?
Train: LightGBM and logistic regression on LEIE labels, data years 2013-2019, NY-only-excluded providers removed.
Test:  New York providers, data years 2020-2022. Positive = provider-year 1-3 years before a NY Medicaid exclusion
       (provider never in LEIE). Negative = NY provider never in LEIE and never on the NY list.
Also reports the same models on LEIE positives in NY for comparison."""
import pandas as pd, numpy as np, glob, zlib
from pandas.api.types import union_categoricals
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import sys
NY, AN = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('data/ny_npi_exclusions.csv', 'data/analysis')   # data_prep/05 output; data_prep/02 output (all negatives)
ny = pd.read_csv(NY, dtype={'NPI': str})
ny_new = ny[~ny.in_leie].groupby('NPI').year.min()
CAT = ['Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type', 'Rndrng_Prvdr_Mdcr_Prtcptg_Ind']
parts = []
for f in sorted(glob.glob(f'{AN}/provider_year_*.parquet')):
    t = pd.read_parquet(f); t = t.drop(columns=[c for c in t.columns if c.startswith('Bene_Race_')])
    isny = t['Rndrng_Prvdr_State_Abrvtn'].astype(str) == 'NY'
    keep = (t.label == 1) | isny | t.Rndrng_NPI.map(lambda n: zlib.crc32(n.encode()) % 4 == 0)
    parts.append(t[keep])
for c in CAT:
    u = union_categoricals([p[c].astype(str).astype('category') for p in parts]).categories
    for p in parts: p[c] = pd.Categorical(p[c].astype(str), categories=u)
d = pd.concat(parts, ignore_index=True); del parts
d['ny_excl_year'] = d.Rndrng_NPI.map(ny_new)
d['ny_pos'] = (d.ny_excl_year - d.year).between(1, 3).astype(int)
isny = d.Rndrng_Prvdr_State_Abrvtn.astype(str) == 'NY'
samp = d.Rndrng_NPI.map(lambda n: zlib.crc32(n.encode()) % 4 == 0)
X = d.drop(columns=['Rndrng_NPI', 'year', 'label', 'years_to_excl', 'ny_excl_year', 'ny_pos'])
NUM = [c for c in X.columns if c not in CAT]
tr = np.where((d.year <= 2019) & d.ny_excl_year.isna() & ((d.label == 1) | samp))[0]
rng = np.random.default_rng(0); neg = tr[d.label.values[tr] == 0]
tr = np.concatenate([tr[d.label.values[tr] == 1], rng.choice(neg, size=len(neg) * 4 // 10, replace=False)])
y = d.label.values
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, reg_lambda=5, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, cat_smooth=50, n_jobs=2, verbose=-1, random_state=0).fit(X.iloc[tr], y[tr], categorical_feature=CAT)
def dense(idx):
    A = np.log1p(np.clip(X.iloc[idx][NUM].fillna(0).values, 0, None)); return A
mu = dense(tr).mean(0); sd = dense(tr).std(0) + 1e-6
lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=500).fit((dense(tr) - mu) / sd, y[tr])
test = (d.year.between(2020, 2022)) & isny
rows = []
for name, pos_col, extra in [('NY Medicaid exclusions (not in LEIE)', 'ny_pos', d.label == 0), ('LEIE fraud exclusions in NY', 'label', d.ny_excl_year.isna())]:
    idx = np.where(test & extra & ((d[pos_col] == 1) | (d.ny_excl_year.isna() & (d.label == 0))))[0]
    yt = d[pos_col].values[idx]
    for mn, s in [('LightGBM', m.predict_proba(X.iloc[idx])[:, 1]), ('LogReg', lr.predict_proba((dense(idx) - mu) / sd)[:, 1])]:
        prev = yt.mean(); ap = average_precision_score(yt, s); o = np.argsort(-s); top = o[:max(1, len(o) // 100)]
        rows.append(dict(label_source=name, model=mn, n_test=len(idx), positives=int(yt.sum()), providers_pos=d.Rndrng_NPI.values[idx][yt == 1].size,
                         prevalence=prev, pr_auc=ap, lift_pr=ap / prev, roc_auc=roc_auc_score(yt, s), lift_top1pct=yt[top].mean() / prev))
R = pd.DataFrame(rows); print(R.round(4).to_string()); R.to_csv('results/ny_external_results.csv', index=False)
