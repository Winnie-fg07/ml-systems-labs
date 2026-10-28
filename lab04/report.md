# Lab 4: Data pipeline and quality

## Problem definition
The goal is to predict whether a person earns more than 50K per year from census records. The inputs are the 14 features of the UCI Adult dataset (age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country and so on). The output is a binary label: income above 50K or not.

Success is measured with accuracy and F1. Accuracy alone is not enough, because only about 23.9% of the people are in the positive class, so a model that always answers "<=50K" would already score about 76% accuracy. F1 shows how well the minority class is found.

There are two constraints. First, there must be no leakage: anything learned from data (imputation values, clipping limits, scaling, encoding) is fitted on the training set only. Second, the pipeline must be reproducible: one command, fixed seed 42, pinned package versions, and a hash of the processed data.

## Goal
Build a leakage-free, reproducible data pipeline for the Adult dataset, validate the raw data, engineer two derived features, audit label quality, and document the result with a processed-data hash, a config file and a data card.

## Method
- Data: OpenML adult, version 2, loaded with fetch_openml. The raw data is never modified. The target column is called "class" in OpenML and is renamed "income".
- Ingestion: 48842 rows and 15 columns were read.
- Validation: schema types, value ranges (age 17-90, hours 1-99, capital-gain 0-99999, capital-loss 0-4356), missing values, exact duplicates and label values. The null policy treats "?" and NaN as missing. Violations are reported, and no rows are dropped during validation.
- Split: 70/30 stratified, random_state=42, done before any cleaning.
- Cleaning (fitted on train only): exact duplicates dropped from train only, test left untouched; median imputation for numeric columns and most-frequent imputation for categorical columns; numeric columns clipped to the 1st and 99th percentile of train; one-hot encoding for categorical columns.
- Scaling: I added StandardScaler for the numeric columns, which the lab does not mention, because logistic regression handles columns on very different scales (such as fnlwgt) badly.
- Derived features: capital_net = capital-gain minus capital-loss, and hours_per_age = hours-per-week / (age - 16). Both use only values from the same row, so they cannot leak information from the test set.
- Model: LogisticRegression (max_iter=1000) trained with and without the derived features.
- The fitted preprocessor is saved with joblib to results/preprocessor.joblib.
- The whole pipeline runs with one command: python lab04\src\lab4.py.

## Results

### Validation report
| Check | Result |
|---|---|
| Types (6 integer, 9 category columns) | PASS, all 15 columns match the schema |
| Ranges (age, hours, capital-gain, capital-loss) | PASS, 0 violations in all four |
| Missing values ("?" or NaN) | workclass 2799, occupation 2809, native-country 857, all other columns 0 (allowed by policy) |
| Exact duplicate rows in raw data | 52 (kept in raw data, reported only) |
| Label values | PASS, only <=50K and >50K |

### Cleaning
- Train rows before cleaning: 34189; test rows: 14653
- Exact duplicates dropped from train: 24; train rows after: 34165
- Processed shapes: train (34165, 107), test (14653, 107)

### Derived features
| Features | Accuracy | F1 |
|---|---|---|
| Original only | 0.8540 | 0.6601 |
| Original + capital_net + hours_per_age | 0.8578 | 0.6696 |

### Label-quality check
- Groups of identical feature rows: 54
- Duplicate-conflict groups (same features, different income): 5, covering 10 rows
- Education check: every education value maps to exactly one education-num
- Manual check of 50 random rows against the data dictionary: no impossible values found (0 of 50). The automatic flags (ranges, education mismatch, relationship versus sex, capital gain and loss both positive) also found 0 problems in the sample.

### Reproducibility
- SHA256 of the processed dataset: 5f8485d44a2784a8f01afc574ee1cd52a90a6227be963d0bc93194bbe6bd46df
- A second run produced the same hash: yes
- Config saved in results/config.yaml

## Three conclusions
1. The validation found no type or range violations, so the raw data matches the schema. The only quality issues are missing values in three columns (workclass, occupation, native-country), 52 exact duplicate rows, and 5 groups of identical records with conflicting labels. These are small compared with 48842 rows, so the data is usable, but the missing values need a proper imputation step.
2. The imputer, clipper and scaler must be fitted on the training set only. If they were fitted on all the data, information from the test set (its medians, percentiles and means) would leak into training, and the test score would look better than it really is. That is why the split comes first and the test set is only transformed.
3. The two derived features raised accuracy from 0.8540 to 0.8578 and F1 from 0.6601 to 0.6696. This is a small improvement from a single split, so I cannot say it is statistically meaningful. It does suggest that net capital is a useful signal for income, because it combines gains and losses into one value.