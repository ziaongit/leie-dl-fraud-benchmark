"""LEIE-Medicare benchmark - official splits and evaluation.

Splits (deterministic, no files needed):
  * provider-grouped: fold of provider NPI in repeat r = crc32(NPI + str(r)) % 5, r = 0, 1, 2  (15 test folds)
  * temporal: train on data years 2013-2019, test on 2020-2022, test set excludes providers positive during training
Rows: all positive provider-years plus all years of the 25% of never-excluded providers with crc32(NPI) % 4 == 0;
negatives carry weight 4 in every metric so that results refer to the full Medicare population.

Usage
  python evaluate.py splits  <data_dir> <phys|partd> <out.csv>           # writes NPI, year, label, weight, fold_r0..r2, temporal_test
  python evaluate.py score   <data_dir> <phys|partd> <predictions.csv>   # predictions: NPI, year, score, split (e.g. grouped_0_3 or temporal)
Prints PR-AUC, lift over the base rate, ROC-AUC, precision and recall among the top 100/500/1,000/5,000 provider-years
(population-weighted), Brier score and expected calibration error (10 equal-width bins), per split and averaged."""
import sys, glob, zlib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def load(data_dir, ds):
    pat, ID = (('provider_year_*.parquet', 'Rndrng_NPI') if ds == 'phys' else ('partd_year_*.parquet', 'NPI'))
    parts = []
    for f in sorted(glob.glob(f'{data_dir}/**/{pat}', recursive=True)):
        t = pd.read_parquet(f, columns=[ID, 'year', 'label']).rename(columns={ID: 'NPI'})
        t = t[(t.label == 1) | t.NPI.map(lambda n: zlib.crc32(n.encode()) % 4 == 0)]
        parts.append(t)
    d = pd.concat(parts, ignore_index=True); d['year'] = d.year.astype(int)
    d['weight'] = np.where(d.label == 1, 1.0, 4.0)
    for r in range(3): d[f'fold_r{r}'] = [zlib.crc32((n + str(r)).encode()) % 5 for n in d.NPI]
    pos_tr = set(d.NPI[(d.year <= 2019) & (d.label == 1)])
    d['temporal_test'] = d.year.between(2020, 2022) & ~d.NPI.isin(pos_tr)
    return d


def metrics(y, s, w):
    prev = (y * w).sum() / w.sum(); o = np.argsort(-s); cw = np.cumsum(w[o]); r = {}
    r['pr_auc'] = average_precision_score(y, s, sample_weight=w); r['lift'] = r['pr_auc'] / prev; r['roc_auc'] = roc_auc_score(y, s, sample_weight=w)
    for k in (100, 500, 1000, 5000):
        sel = o[:np.searchsorted(cw, k) + 1]; tp = (y[sel] * w[sel]).sum()
        r[f'prec@{k}'] = tp / w[sel].sum(); r[f'recall@{k}'] = tp / (y * w).sum()
    r['brier'] = (w * (s - y) ** 2).sum() / w.sum()
    bins = np.minimum((s * 10).astype(int), 9); e = 0               # 10 equal-width bins, as in the paper
    for k in range(10):
        m = bins == k
        if m.any(): e += w[m].sum() / w.sum() * abs((y[m] * w[m]).sum() / w[m].sum() - (s[m] * w[m]).sum() / w[m].sum())
    r['ece'] = e
    return r


def test_rows(d, split):
    if split == 'temporal': return d.temporal_test.values
    _, r, f = split.split('_'); return (d[f'fold_r{r}'] == int(f)).values


if __name__ == '__main__':
    cmd, data_dir, ds = sys.argv[1], sys.argv[2], sys.argv[3]
    d = load(data_dir, ds)
    if cmd == 'splits':
        d.to_csv(sys.argv[4], index=False); print('rows', len(d), 'positives', int(d.label.sum()))
        print(d.groupby('fold_r0').label.agg(['size', 'sum']).to_string())
    else:
        p = pd.read_csv(sys.argv[4], dtype={'NPI': str}); out = []
        for split, q in p.groupby('split'):
            t = d[test_rows(d, split)].merge(q[['NPI', 'year', 'score']], on=['NPI', 'year'], how='left')
            if t.score.isna().any(): sys.exit(f'{split}: {int(t.score.isna().sum())} test rows have no score')
            r = metrics(t.label.values, t.score.values, t.weight.values); r['split'] = split; out.append(r)
        R = pd.DataFrame(out).set_index('split'); print(R.round(5).to_string()); print('\nmean over splits'); print(R.mean().round(5).to_string())
