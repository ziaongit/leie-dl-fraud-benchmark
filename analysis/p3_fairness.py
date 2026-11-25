"""Who gets flagged: share of rural providers and of each provider type among the top-1,000 provider-years of each model
(grouped design, all 15 folds, original feature order), compared with the test population (weighted).
Usage: python p3_fairness.py <results_dir> <dataset>"""
import sys, glob, os, numpy as np, pandas as pd
RD, DS = sys.argv[1], sys.argv[2]
pat, RU, TY = (('phys/provider_year_*.parquet', 'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_Type') if DS == 'phys' else ('partd/partd_year_*.parquet', 'cat_RUCA', 'cat_Type'))
d = pd.concat([pd.read_parquet(f, columns=['label', RU, TY]) for f in sorted(glob.glob(pat))], ignore_index=True)
ru = pd.to_numeric(d[RU].astype(str), errors='coerce').values; rural = (ru >= 4).astype(float); rural[np.isnan(ru)] = np.nan
ty = d[TY].astype(str).values; y_all = d.label.values
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']; rows = []; types = []
for rep in range(3):
    for fold in range(5):
        key = f'{DS}_grouped_{rep}_{fold}'; te = np.load(f'{RD}/scores/{key}_test_idx.npy'); w = np.where(y_all[te] == 1, 1.0, 4.0)
        pop_r = np.nansum(rural[te] * w) / np.sum(w[~np.isnan(rural[te])])
        pt = pd.Series(w, index=ty[te]).groupby(level=0).sum() / w.sum()
        for m in M:
            s = np.load(f'{RD}/scores/{key}_{m}_0.npy'); o = np.argsort(-s); cw = np.cumsum(w[o]); top = o[:np.searchsorted(cw, 1000) + 1]
            ft = pd.Series(w[top], index=ty[te][top]).groupby(level=0).sum() / w[top].sum()
            r_top = np.nansum(rural[te][top] * w[top]) / np.sum(w[top][~np.isnan(rural[te][top])])
            rows.append(dict(dataset=DS, rep=rep, fold=fold, model=m, rural_pop=pop_r, rural_top=r_top))
            for t_, v in ft.items(): types.append(dict(model=m, rep=rep, fold=fold, type=t_, top=v, pop=pt.get(t_, 0)))
R = pd.DataFrame(rows); T = pd.DataFrame(types)
R.to_csv(f'{RD}/p3_fairness_rural_{DS}.csv', index=False); T.to_csv(f'{RD}/p3_fairness_types_{DS}.csv', index=False)
print(R.groupby('model')[['rural_pop', 'rural_top']].mean().round(3).to_string())
for m in ['LightGBM', 'AttnCNNRNN', 'MLP']:
    g = T[T.model == m].groupby('type')[['top', 'pop']].sum() / 15; g['ratio'] = g.top / g['pop']
    print(m, g.sort_values('top', ascending=False).head(6).round(3).to_string())
