"""Build LEADERBOARD.md from the benchmark result files (original feature order, all repeats/folds)."""
import pandas as pd, sys
NM = {'LogReg': 'Logistic regression', 'LightGBM': 'LightGBM', 'XGBoost': 'XGBoost', 'MLP': 'MLP', 'FTT': 'FT-Transformer', 'CNNRNN': 'CNN-RNN', 'AttnCNNRNN': 'Attention CNN-RNN'}
out = ['# LEIE-Medicare benchmark - leaderboard', '',
       'Confirmed fraud labels (OIG LEIE fraud-related exclusions), natural prevalence, population-weighted metrics.',
       'Grouped: provider-grouped 5-fold CV x 3 repeats (15 test folds); temporal: train 2013-2019, test 2020-2022 (3 training runs).',
       'Evaluate a new model with `python benchmark/evaluate.py score ...` on the same splits and open a pull request to add a row.', '']
for ds, dn in [('phys', 'Physician and Other Practitioners (2013-2024)'), ('partd', 'Part D Prescribers (2013-2023)')]:
    d = pd.read_csv(f'results/run1/results_{ds}.csv'); d = d[d.perm == 0]
    for des in ['grouped', 'temporal']:
        x = d[d.design == des].groupby('model').agg(pr=('pr_auc', 'mean'), lm=('lift_pr', 'mean'), lmed=('lift_pr', 'median'), roc=('roc_auc', 'mean'), p1k=('prec_at_1000', 'mean'), n=('pr_auc', 'size'))
        x = x.sort_values('lmed', ascending=False)
        out += [f'## {dn} - {des}', '', '| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |', '|---|---|---|---|---|---|---|']
        for i, (m, r) in enumerate(x.iterrows(), 1):
            out.append(f'| {i} | {NM[m]} | {r.pr:.4f} | {r.lm:.1f} / {r.lmed:.1f} | {r.roc:.3f} | {100 * r.p1k:.2f} | {int(r.n)} |')
        out.append('')
out += ['Ranked by median PR-AUC lift over the base rate. Source: Ullah and Khan (manuscript, 2026), run 1 of this repository.', '']
open('LEADERBOARD.md', 'w').write('\n'.join(out)); print('\n'.join(out[:22]))
