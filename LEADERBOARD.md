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

## Physician - provider histories (grouped repeat 0, 5 folds)

Input: the provider's numeric profile in years t-3 to t (see `history/p3_hist.py`).

| Rank | Model | PR-AUC | Lift (mean / median) | ROC-AUC | Precision@1,000 (%) | Splits |
|---|---|---|---|---|---|---|
| 1 | LightGBM (4-year history, flattened) | 0.0048 | 46.5 / 38.4 | 0.842 | 1.16 | 5 |
| 2 | Attention GRU over years | 0.0048 | 45.7 / 26.2 | 0.873 | 0.76 | 5 |
| 3 | GRU over years | 0.0036 | 32.3 / 26.1 | 0.874 | 1.24 | 5 |
| 4 | MLP (4-year history, flattened) | 0.0023 | 21.7 / 19.5 | 0.866 | 0.96 | 5 |
| 5 | LightGBM (year t only) | 0.0015 | 14.0 / 13.5 | 0.830 | 0.60 | 5 |

Ranked by median PR-AUC lift over the base rate. Source: Ullah and Khan (manuscript, 2026), run 1 of this repository.
