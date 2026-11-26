# LEIE-Medicare benchmark - leaderboard

Confirmed fraud labels (OIG LEIE fraud-related exclusions), natural prevalence, population-weighted metrics.
Grouped: provider-grouped 5-fold CV x 3 repeats (15 test folds); temporal: train 2013-2019, test 2020-2022 (3 training runs).
Evaluate a new model with `python benchmark/evaluate.py score ...` on the same splits and open a pull request to add a row.

## Physician and Other Practitioners (2013-2024) - grouped

| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.0022 | 19.6 / 14.1 | 0.831 | 0.57 | 15 |
| 2 | XGBoost | 0.0016 | 15.0 / 13.4 | 0.820 | 0.71 | 15 |
| 3 | MLP | 0.0012 | 11.7 / 10.8 | 0.838 | 0.41 | 15 |
| 4 | Logistic regression | 0.0022 | 21.7 / 9.5 | 0.846 | 0.21 | 15 |
| 5 | Attention CNN-RNN | 0.0010 | 9.5 / 9.0 | 0.822 | 0.35 | 15 |
| 6 | FT-Transformer | 0.0012 | 11.5 / 8.1 | 0.825 | 0.33 | 15 |
| 7 | CNN-RNN | 0.0009 | 8.3 / 7.2 | 0.799 | 0.35 | 15 |

## Physician and Other Practitioners (2013-2024) - temporal

| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.0007 | 8.0 / 8.0 | 0.803 | 0.33 | 3 |
| 2 | MLP | 0.0007 | 7.7 / 7.9 | 0.813 | 0.20 | 3 |
| 3 | XGBoost | 0.0007 | 8.0 / 7.5 | 0.809 | 0.40 | 3 |
| 4 | Logistic regression | 0.0006 | 6.5 / 6.7 | 0.830 | 0.20 | 3 |
| 5 | CNN-RNN | 0.0005 | 5.7 / 5.5 | 0.800 | 0.13 | 3 |
| 6 | Attention CNN-RNN | 0.0005 | 5.6 / 5.1 | 0.799 | 0.17 | 3 |
| 7 | FT-Transformer | 0.0004 | 4.4 / 4.1 | 0.786 | 0.10 | 3 |

## Part D Prescribers (2013-2023) - grouped

| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.0030 | 17.8 / 12.4 | 0.821 | 0.99 | 15 |
| 2 | XGBoost | 0.0024 | 14.5 / 10.7 | 0.809 | 0.87 | 15 |
| 3 | MLP | 0.0017 | 10.0 / 9.0 | 0.807 | 0.69 | 15 |
| 4 | Logistic regression | 0.0012 | 7.2 / 7.3 | 0.822 | 0.39 | 15 |
| 5 | CNN-RNN | 0.0014 | 8.3 / 6.5 | 0.805 | 0.48 | 15 |
| 6 | Attention CNN-RNN | 0.0012 | 7.2 / 6.0 | 0.791 | 0.35 | 15 |
| 7 | FT-Transformer | 0.0012 | 6.9 / 5.7 | 0.800 | 0.29 | 15 |

## Part D Prescribers (2013-2023) - temporal

| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.0008 | 6.7 / 6.8 | 0.790 | 0.27 | 3 |
| 2 | MLP | 0.0007 | 6.6 / 6.7 | 0.783 | 0.27 | 3 |
| 3 | XGBoost | 0.0007 | 6.2 / 6.2 | 0.781 | 0.30 | 3 |
| 4 | Logistic regression | 0.0007 | 6.1 / 6.1 | 0.794 | 0.27 | 3 |
| 5 | FT-Transformer | 0.0006 | 5.1 / 5.3 | 0.777 | 0.30 | 3 |
| 6 | Attention CNN-RNN | 0.0007 | 6.0 / 5.3 | 0.767 | 0.40 | 3 |
| 7 | CNN-RNN | 0.0005 | 4.1 / 3.9 | 0.729 | 0.10 | 3 |

## Provider histories (genuine sequences), grouped design, 15 folds per data source

Input: the provider's numeric profile in years t-3 to t (see `history/p3_hist.py`). Lift: PR-AUC lift over the base rate (mean / median); P@1000: precision among the top 1,000 provider-years (%).

| Rank | Model | Physician lift | Physician ROC-AUC | Physician P@1000 | Part D lift | Part D ROC-AUC | Part D P@1000 |
|---|---|---|---|---|---|---|---|
| 1 | LightGBM (4-year history, flattened) | 50.2 / 38.4 | 0.858 | 1.17 | 61.2 / 55.7 | 0.858 | 2.68 |
| 2 | GRU over years | 38.2 / 36.1 | 0.868 | 1.26 | 41.3 / 38.6 | 0.856 | 2.20 |
| 3 | MLP (4-year history, flattened) | 26.3 / 19.6 | 0.867 | 0.99 | 36.1 / 36.5 | 0.851 | 2.02 |
| 4 | Attention GRU over years | 36.0 / 21.0 | 0.863 | 0.81 | 35.4 / 30.7 | 0.854 | 1.97 |
| 5 | LightGBM (year t only) | 19.6 / 14.1 | 0.831 | 0.57 | 17.9 / 12.4 | 0.821 | 0.99 |

Ranked by median lift (mean of the two data sources). Source: runs 5 and 6, `results/run6/hist_15fold.csv`.

## Independent labels: 32 state Medicaid exclusion lists (test 2020-2022)

Models trained on LEIE labels (2013-2019) and tested against state exclusions of providers never in the LEIE (343 Physician and 428 Part D positive provider-years). Provider-level bootstrap 95% intervals (2,000 replicates). Source: run 7, `results/run7/p3_states_ci.csv`.

| Model | Physician ROC-AUC | Physician lift | Part D ROC-AUC | Part D lift |
|---|---|---|---|---|
| LightGBM | 0.72 (0.68-0.76) | 5.7 (3.7-11.0) | 0.70 (0.67-0.74) | 9.4 (3.1-35.8) |
| CNN-RNN | 0.71 (0.67-0.75) | 5.9 (2.8-26.4) | 0.70 (0.67-0.73) | 4.0 (2.7-6.6) |
| XGBoost | 0.69 (0.65-0.73) | 4.5 (2.9-8.0) | 0.71 (0.67-0.74) | 4.7 (2.9-9.4) |
| MLP | 0.68 (0.64-0.72) | 4.2 (2.7-9.5) | 0.73 (0.70-0.76) | 3.5 (2.6-5.6) |
| FT-Transformer | 0.69 (0.65-0.73) | 3.3 (2.4-5.0) | 0.72 (0.68-0.75) | 3.0 (2.3-4.3) |
| Attention CNN-RNN | 0.64 (0.60-0.68) | 3.2 (1.6-14.6) | 0.69 (0.66-0.72) | 2.5 (2.0-3.4) |
| Logistic regression | 0.70 (0.67-0.74) | 2.8 (2.2-4.0) | 0.72 (0.69-0.75) | 3.4 (2.5-4.9) |

## Retrospective deployment simulation (deployment years 2016-2022)

Retrained each year on the exclusions known at that time; the top 1,000 / 5,000 provider-years of the next data year are audited. Random audits of 1,000 would find 0.11 (Physician) and 0.17 (Part D) cases per year. Source: `results/deployment/`.

| Model | Physician: cases/yr at 1,000 | at 5,000 | Part D: cases/yr at 1,000 | at 5,000 |
|---|---|---|---|---|
| XGBoost | 3.3 | 9.1 | 4.4 | 12.3 |
| LightGBM | 2.7 | 10.3 | 4.0 | 13.0 |
| MLP | 2.0 | 8.9 | 3.4 | 12.4 |
| Logistic regression | 1.7 | 6.7 | 2.9 | 9.7 |
