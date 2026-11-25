# Sequence and attention deep learning vs tree ensembles on confirmed Medicare provider fraud

Code and result files for the paper

> Ullah Z, Khan W. *Do Sequence and Attention Deep Learning Models Help Detect Confirmed Medicare Provider Fraud? A Leakage-Free Benchmark on Exclusion-List Labels.* Manuscript, 2026.

The study links public CMS Medicare data to **confirmed fraud labels** (federal exclusions in the OIG List of Excluded Individuals/Entities, LEIE) and compares seven models under leakage-free evaluation:
logistic regression, LightGBM, XGBoost, an MLP, an FT-Transformer, a CNN-RNN and an attention CNN-RNN.

## What is in this repository

| Folder | Content |
|---|---|
| `data_prep/` | Build the labelled provider-year datasets from the public CMS and LEIE files |
| `benchmark/` | `p3_bench.py` (main benchmark, all models) and `p3_attention.py` (attention vs SHAP, permutation and ablation) |
| `analysis/` | Summary tables, corrected t-tests, equivalence tests, complementarity, combination, confusion counts, calibration, figures |
| `robustness/` | Label-noise simulation and the independent New York State exclusion-list test: `p3_robust_deep.py` (all seven models, full tuning; used in the paper) and the earlier fixed-configuration versions `label_noise.py`, `ny_external.py` |
| `kaggle/` | The exact Kaggle notebook cells used for the two GPU runs |
| `results/` | All result files reported in the paper (per-split metrics, statistics, attention results, learning curve) |

Raw data, processed parquet files and saved test scores are **not** included (size); every one of them can be rebuilt with the scripts below.

## Data (all public)

| Source | Version used | Where |
|---|---|---|
| CMS Medicare Physician & Other Practitioners - by Provider | data years 2013-2024 | data.cms.gov |
| CMS Medicare Part D Prescribers - by Provider | data years 2013-2023 | data.cms.gov |
| OIG LEIE downloadable database (`UPDATED.csv`) | update of October 2026, downloaded November 2026 | oig.hhs.gov/exclusions |
| New York OMIG Medicaid exclusions (OpenSanctions dataset `us_ny_med_exclusions`, FollowTheMoney JSON) | release of 16 November 2026 | opensanctions.org |

## Labels

* **Positive:** provider-year 1-3 years before a fraud-related LEIE exclusion (sections 1128(a)(1), 1128(a)(3), 1128(b)(7)), matched by NPI.
* **Negative:** provider never appears in the LEIE (any type, any date).
* **Removed:** provider-years in or after the exclusion year, fraud exclusions more than 3 years later, and providers with other exclusion types.
* All positives and all years of a 25% sample of never-excluded providers (`crc32(NPI) % 4 == 0`) are kept; negatives carry weight 4 in every metric so results refer to the full population.

No feature is derived from the LEIE or from the label. Names, addresses and beneficiary race variables are dropped.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Physician data: one labelled file per data year, then assemble and sample
for Y in 2013 2014 ... 2024; do
  python data_prep/01_build_provider_year.py <cms_physician_by_provider_$Y.csv> $Y <LEIE.csv> data/provider_year
done
python data_prep/02_assemble_physician.py data/provider_year data/analysis
python data_prep/03_sample_physician.py data/analysis data/phys

# 2. Part D data (all years in one folder; data year read from the file name)
python data_prep/04_build_partd.py <folder_with_partd_by_provider_csvs> <LEIE.csv> data/partd

# 3. New York exclusion list (for the external-label test)
python data_prep/05_ny_exclusions.py <us_ny_med_exclusions.json> <LEIE.csv> data/ny_npi_exclusions.csv data/provider_year

# 4. Benchmark (GPU recommended; on Kaggle see kaggle/run_kaggle.py)
DATA=data WORK=out python benchmark/p3_bench.py phys grouped 3 5
DATA=data WORK=out python benchmark/p3_bench.py phys temporal 3
DATA=data WORK=out python benchmark/p3_bench.py partd grouped 3 5
DATA=data WORK=out python benchmark/p3_bench.py partd temporal 3

# 5. Attention faithfulness and learning curve
DATA=data WORK=out ATT_FOLDS=5 python benchmark/p3_attention.py phys     # needs benchmark/p3_bench.py in the same folder
DATA=data WORK=out DATA_FRAC=0.25 PERMS=1 python benchmark/p3_bench.py phys grouped 1 5   # also 0.1 and 0.5

# 6. Analyses
python analysis/p3_analyse.py out                      # summary, Nadeau-Bengio + Holm tests, order sensitivity
python analysis/p3_labels.py out data                  # rebuild test labels for the saved scores
python analysis/p3_extra.py out                        # TOST equivalence, complementarity, combination
python analysis/p3_confcal.py out phys                 # confusion counts at k = 1,000 / 5,000 and calibration bins
python analysis/p3_censor.py out data phys             # sensitivity: drop data years with incomplete LEIE look-ahead (>= 2023)
DATA=data WORK=out python robustness/p3_robust_deep.py noise   # label noise, all 7 models (needs benchmark/p3_bench.py alongside)
DATA=data WORK=out python robustness/p3_robust_deep.py ny      # New York independent labels, all 7 models
RUN1=out RUN2=out python analysis/make_figs.py && RUN1=out RUN2=out python analysis/make_figs2.py
```

## Design choices (leakage-free evaluation)

* Provider-grouped 5-fold cross-validation (fold = hash of NPI), repeated 3 times, plus a temporal split (train 2013-2019, test 2020-2022; providers positive during training removed from the test set).
* All preprocessing (log transform, standardisation, clipping at 8 SD, category vocabularies with >= 20 occurrences) is fitted on the training rows of each split only. No SMOTE or other resampling; class weights in every model.
* Equal tuning budget: 6 random configurations per model and split, selected by PR-AUC on an inner provider-grouped validation set (15% of training providers). Search spaces are in `benchmark/p3_bench.py`; the selected configuration of every split is stored in the `cfg` column of `results/run1/results_*.csv`.
* Neural networks: AdamW, batch 2,048, weighted binary cross-entropy, up to 10 epochs, early stopping (patience 3). All models are trained from scratch; no pretrained weights.
* Sequence models are trained under the original column order and 2 random permutations (seeds 77 and 78) on the first repeat.
* Random seeds are fixed in the code (fold assignment by `crc32`, model seeds by repeat/fold).

## Result files

| File | Content |
|---|---|
| `results/run1/results_phys.csv`, `results_partd.csv` | One row per dataset, design, repeat, fold, model and feature order: PR-AUC, ROC-AUC, lift, precision/recall at k, Brier, ECE, selected configuration, training time |
| `results/run1/p3_summary.csv`, `p3_pairwise.csv`, `p3_equivalence.csv` | Means, corrected tests with Holm adjustment and effect sizes, TOST equivalence |
| `results/run1/p3_order_sensitivity.csv` | Range of PR-AUC and ROC-AUC across feature orders |
| `results/run1/p3_complementarity.csv`, `p3_combination.csv` | Overlap of top-1,000 lists, rank-average combinations |
| `results/run1/p3_confusion_*.csv`, `p3_calibration_*.csv` | Confusion counts at audit budgets, reliability-diagram bins |
| `results/run1/p3_censor_*.csv` | Grouped results recomputed without data years 2023-2024 (right-censored labels) |
| `results/run2/p3_attention_results_*.csv`, `p3_attention_importances_*.csv` | Attention vs SHAP, permutation and ablation importance (per fold, order and feature) |
| `results/run2/results_phys_lc*.csv` | Learning curve (10%, 25%, 50% of training providers) |
| `results/run3/p3_robust_noise.csv` | Label noise: 5 folds x q in {0, 0.25, 0.5} x 7 models (tuned) |
| `results/run3/p3_robust_ny.csv` | New York independent labels (107 positives) and LEIE-in-New-York comparison (27 positives), 7 models |
| `results/label_noise_results.csv`, `ny_external_results.csv` | Earlier fixed-configuration robustness runs (4 and 2 models; superseded by run 3) |

Note: run 1 used `p3_bench.py` without the `DATA_FRAC` option; the version here adds only that option (default 1.0 = identical behaviour).

## Changelog

* 2026-11-25: added run 3 (`robustness/p3_robust_deep.py`, `results/run3/`): label-noise and New York analyses for all seven models with full tuning, as reported in paper draft v0.7.
* 2026-11-25: fixed the Nadeau-Bengio correction in `p3_analyse.py` and `p3_extra.py` (the ratio n_test/n_train now uses the fold sizes, 1/(K-1) = 0.25; the first version used the sub-sampled training size, which made the tests too conservative). `p3_pairwise.csv`, `p3_equivalence.csv` and `p3_summary.csv` were regenerated. Added `p3_censor.py`.

## Licence and citation

Code: MIT licence. Data: public U.S. government data; please cite CMS, OIG and OpenSanctions as data sources.
If you use this code, please cite the paper above.
