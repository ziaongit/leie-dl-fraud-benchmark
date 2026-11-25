"""Paper 3 v2 - analysis of benchmark results.
 - mean (SD) of every metric per dataset x design x model (original feature order, perm 0)
 - pairwise comparisons vs the best model with the Nadeau-Bengio corrected resampled t-test (grouped CV),
   Holm correction across comparisons, and effect sizes (mean difference, Cohen's d of paired differences)
 - feature-order sensitivity of CNN-RNN models: spread of PR-AUC/ROC-AUC across random feature orders on the same folds
Writes p3_summary.csv, p3_pairwise.csv, p3_order_sensitivity.csv."""
import pandas as pd, numpy as np, glob
import sys
RD = sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working'
from scipy import stats
R = pd.concat([pd.read_csv(f) for f in glob.glob(RD + '/results_*.csv')], ignore_index=True)
MET = ['pr_auc', 'roc_auc', 'lift_pr', 'prec_at_100', 'prec_at_1000', 'recall_at_1000', 'recall_at_5000', 'brier', 'ece', 'secs']
base = R[R.perm == 0]
S = base.groupby(['dataset', 'design', 'model'])[MET].agg(['mean', 'std', 'count'])
S.columns = ['_'.join(c) for c in S.columns]; S = S.reset_index(); S.to_csv(RD + '/p3_summary.csv', index=False)

def nb_ttest(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    J = len(diff); m = diff.mean(); v = diff.var(ddof=1)
    if J < 2 or v == 0: return np.nan, np.nan
    t = m / np.sqrt((1 / J + n_test / n_train) * v); p = 2 * stats.t.sf(abs(t), J - 1); return t, p

rows = []
for (ds, des), sub in base.groupby(['dataset', 'design']):
    for metric in ['pr_auc', 'roc_auc', 'prec_at_1000']:
        piv = sub.pivot_table(index=['rep', 'fold'], columns='model', values=metric)
        best = piv.mean().idxmax()
        ntr = sub.n_train.mean(); nte = sub.n_test.mean()
        for mdl in piv.columns:
            if mdl == best: continue
            dfr = (piv[best] - piv[mdl]).dropna().values
            if des == 'grouped': t, p = nb_ttest(dfr, ntr, nte)
            else: t, p = (stats.ttest_1samp(dfr, 0) if len(dfr) > 1 else (np.nan, np.nan))
            d = dfr.mean() / dfr.std(ddof=1) if len(dfr) > 1 and dfr.std(ddof=1) > 0 else np.nan
            rows.append(dict(dataset=ds, design=des, metric=metric, best=best, vs=mdl, n=len(dfr), mean_diff=dfr.mean(),
                             rel_diff=dfr.mean() / piv[mdl].mean(), wins=int((dfr > 0).sum()), t=t, p=p, cohen_d=d))
P = pd.DataFrame(rows)
if len(P):
    P['p_holm'] = np.nan
    for k, g in P.groupby(['dataset', 'design', 'metric']):
        ps = g.p.fillna(1).values; o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
        for i, j in enumerate(o): run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
        P.loc[g.index, 'p_holm'] = adj
P.to_csv(RD + '/p3_pairwise.csv', index=False)

seq = R[R.model.isin(['CNNRNN', 'AttnCNNRNN'])]
O = seq.groupby(['dataset', 'design', 'model', 'rep', 'fold']).agg(n_orders=('perm', 'nunique'), pr_min=('pr_auc', 'min'), pr_max=('pr_auc', 'max'),
                                                                   roc_min=('roc_auc', 'min'), roc_max=('roc_auc', 'max')).reset_index()
O = O[O.n_orders > 1]; O['pr_range_rel'] = (O.pr_max - O.pr_min) / O.pr_min; O['roc_range'] = O.roc_max - O.roc_min
O.to_csv(RD + '/p3_order_sensitivity.csv', index=False)
print(S[['dataset', 'design', 'model', 'pr_auc_mean', 'roc_auc_mean', 'prec_at_1000_mean', 'pr_auc_count']].round(4).to_string())
print(P.round(4).to_string() if len(P) else 'no pairwise yet'); print(O.round(3).to_string())
