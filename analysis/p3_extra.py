"""Paper 3 v2 - extra analyses on saved test scores (results from p3_bench.py / Kaggle).
1. Equivalence (TOST): is each deep model practically equivalent to the best tree model?
   Margin = 10% of the best model's mean PR-AUC (relative) and 0.01 ROC-AUC; Nadeau-Bengio corrected variance.
2. Complementarity: overlap (Jaccard) of the top-1000 ranked providers between models; share of
   true positives found by a deep model but missed by LightGBM in its top-1000.
3. Simple combination: rank-average of LightGBM with each deep model (and with all) - does it beat LightGBM alone?
Usage: python p3_extra.py <results_dir>   (folder with results_*.csv and scores/)"""
import sys, os, glob, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
RD = sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working'
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{RD}/results_*.csv')], ignore_index=True); R = R[R.perm == 0]
DEEP = ['MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']

def nb_var(diff, ntr, nte): return (1 / len(diff) + nte / ntr) * diff.var(ddof=1)
rows = []
for (ds, des), sub in R.groupby(['dataset', 'design']):
    for metric, margin_fn in [('pr_auc', lambda b: 0.10 * b), ('roc_auc', lambda b: 0.01)]:
        piv = sub.pivot_table(index=['rep', 'fold'], columns='model', values=metric)
        trees = [m for m in ['LightGBM', 'XGBoost'] if m in piv]; best = piv[trees].mean().idxmax(); M = margin_fn(piv[best].mean())
        for mdl in [m for m in DEEP + ['LogReg'] if m in piv]:
            dif = (piv[mdl] - piv[best]).dropna().values; J = len(dif)
            if J < 2: continue
            se = np.sqrt(nb_var(dif, sub.n_train.mean(), sub.n_test.mean())) if des == 'grouped' else dif.std(ddof=1) / np.sqrt(J)
            t1 = (dif.mean() + M) / se; t2 = (dif.mean() - M) / se
            p_tost = max(stats.t.sf(t1, J - 1), stats.t.cdf(t2, J - 1))
            rows.append(dict(dataset=ds, design=des, metric=metric, reference=best, model=mdl, mean_diff=dif.mean(), margin=M,
                             ci90_lo=dif.mean() - stats.t.ppf(0.95, J - 1) * se, ci90_hi=dif.mean() + stats.t.ppf(0.95, J - 1) * se,
                             p_equivalence=p_tost, equivalent=p_tost < 0.05, worse=(dif.mean() + stats.t.ppf(0.95, J - 1) * se) < -M))
pd.DataFrame(rows).to_csv(f'{RD}/p3_equivalence.csv', index=False)

# complementarity and combination need labels of the test rows
comp, comb = [], []
lab = {}
for f in glob.glob(f'{RD}/scores/*_test_idx.npy'):
    pass  # labels are written by p3_labels below
lab_file = f'{RD}/test_labels.npz'
if os.path.exists(lab_file):
    L = np.load(lab_file, allow_pickle=True)
    for f in sorted(glob.glob(f'{RD}/scores/*_LightGBM_0.npy')):
        key = os.path.basename(f).replace('_LightGBM_0.npy', '')            # e.g. phys_grouped_0_3
        if key + '_y' not in L: continue
        yt, wt = L[key + '_y'], L[key + '_w']; sL = np.load(f)
        def topk(s, k=1000):
            o = np.argsort(-s); cut = np.searchsorted(np.cumsum(wt[o]), k); return set(o[:cut + 1])
        TL = topk(sL); posL = {i for i in TL if yt[i] == 1}
        allr = [rankdata(sL)]
        for mdl in DEEP:
            fm = f'{RD}/scores/{key}_{mdl}_0.npy'
            if not os.path.exists(fm): continue
            sM = np.load(fm); TM = topk(sM); posM = {i for i in TM if yt[i] == 1}; allr.append(rankdata(sM))
            comp.append(dict(split=key, model=mdl, jaccard_top1000=len(TL & TM) / len(TL | TM), tp_lgbm=len(posL), tp_model=len(posM),
                             tp_only_model=len(posM - posL), tp_union=len(posL | posM)))
            sc = (rankdata(sL) + rankdata(sM)) / 2
            comb.append(dict(split=key, combo='LightGBM+' + mdl, pr_auc=average_precision_score(yt, sc, sample_weight=wt),
                             roc_auc=roc_auc_score(yt, sc, sample_weight=wt), pr_auc_lgbm=average_precision_score(yt, sL, sample_weight=wt)))
        if len(allr) > 2:
            sc = np.mean(allr, 0)
            comb.append(dict(split=key, combo='LightGBM+all deep', pr_auc=average_precision_score(yt, sc, sample_weight=wt),
                             roc_auc=roc_auc_score(yt, sc, sample_weight=wt), pr_auc_lgbm=average_precision_score(yt, sL, sample_weight=wt)))
    pd.DataFrame(comp).to_csv(f'{RD}/p3_complementarity.csv', index=False); pd.DataFrame(comb).to_csv(f'{RD}/p3_combination.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string())
if comp: print(pd.DataFrame(comp).groupby('model').mean(numeric_only=True).round(3).to_string())
if comb: print(pd.DataFrame(comb).groupby('combo').mean(numeric_only=True).round(5).to_string())
