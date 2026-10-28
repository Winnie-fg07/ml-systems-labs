# Data card: UCI Adult (Census Income)

## Source
The Adult dataset comes from the 1994 US Census database and is hosted by the UCI Machine Learning Repository (University of California, Irvine). In this project it is loaded through OpenML (adult, version 2). The task is to predict whether a person earns more than 50K per year.

## License
CC BY 4.0

## Size
- 48842 rows and 15 columns (14 features and 1 label).
- Train set: 34165 rows after removing 24 exact duplicates; test set: 14653 rows (untouched).
- After one-hot encoding and the derived features the processed data has 107 columns.

## Processing
Split 70/30 (stratified, seed 42) before cleaning. Median and most-frequent imputation, clipping at the 1st and 99th percentile, scaling and one-hot encoding are all fitted on the train set only. Processed-data hash (SHA256): 5f8485d44a2784a8f01afc574ee1cd52a90a6227be963d0bc93194bbe6bd46df.

## Known biases
- The data is from 1994, so income levels, jobs and social patterns do not match today.
- Class imbalance: only 23.9% of the people earn more than 50K.
- Group imbalance: 66.9% male and 33.1% female. Race is 85.5% White, 9.6% Black, 3.1% Asian-Pac-Islander, 1.0% Amer-Indian-Eskimo and 0.8% Other. 89.7% have native-country United-States, so other countries are poorly represented.
- The label is a threshold on income, and 50K in 1994 is not the same as 50K today.
- fnlwgt is a survey sampling weight, not a personal characteristic, and it is used here as a normal feature.
- A model trained on this data can learn and repeat historical differences between groups (for example by sex or race), so its predictions should not be used for real decisions about people without a fairness review.
- Missing values are concentrated in workclass (2799), occupation (2809) and native-country (857), and they may not be missing at random.

## Privacy concerns
The data holds age, sex, race, marital status, occupation and native country. Combinations of these can make some individuals identifiable, especially in small groups. There are no names or IDs, and the data is public, but it should not be linked with other datasets to identify people. Sensitive attributes such as race and sex should only be used with care.

## Label quality
5 groups of identical records (10 rows) have conflicting income labels, which shows some label noise. A manual check of 50 random rows against the data dictionary found no impossible values.

## Intended use and limits
Suitable for teaching and for practising data pipelines. Not suitable for real decisions about hiring, credit or similar.