# Titanic Data Science Project

## Project Overview

The **Titanic Data Science Project** is an end-to-end data science project based on the Kaggle **Titanic - Machine Learning from Disaster** dataset.

The project was developed progressively across multiple internship tasks and currently covers three major stages:

- **Week 1 – Data Cleaning and Preprocessing**
- **Week 2 – Exploratory Data Analysis and Visualization**
- **Week 4 – Supervised Learning and Survival Prediction**

Instead of treating each task as an unrelated project, the same Titanic dataset is developed progressively from raw-data preparation to exploratory analysis and finally predictive modeling.

---

# Overall Project Workflow

```text
Raw Titanic Dataset
        │
        ▼
Week 1
Data Acquisition
        │
        ▼
Initial Exploration
        │
        ▼
Missing-Value Analysis
        │
        ▼
Outlier and Consistency Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Cleaned Titanic Dataset
        │
        ▼
Week 2
Dataset Overview
        │
        ▼
Univariate Analysis
        │
        ▼
Bivariate Analysis
        │
        ▼
Multivariate Analysis
        │
        ▼
Correlation Analysis
        │
        ▼
Interpretation and Insights
        │
        ▼
Week 4
Supervised Learning
        │
        ▼
80/20 Stratified Train/Test Split
        │
        ▼
Leakage-Safe Preprocessing
        │
        ▼
5-Fold Stratified Cross-Validation
        │
        ▼
Baseline Model Comparison
        │
        ├── Logistic Regression
        ├── Decision Tree
        └── Random Forest
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Random Forest Selection
        │
        ▼
Final Holdout Evaluation
        │
        ▼
Feature Importance
        │
        ▼
Error and Subgroup Analysis
        │
        ▼
Feature Engineering Evaluation
        │
        ▼
Probability Calibration
```

---

# Dataset Information

The project uses the Titanic dataset from Kaggle's **Titanic - Machine Learning from Disaster** competition.

The primary dataset is:

```text
data/train.csv
```

The repository also contains:

```text
data/test.csv
```

The project primarily uses `train.csv` because it contains the target variable `Survived`, which is required for exploratory survival analysis and supervised model training.

## Original Dataset Shape

- **Rows:** 891
- **Columns:** 12

## Original Columns

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Survival status: 0 = No, 1 = Yes |
| `Pclass` | Passenger class |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation |

---

# Week 1 — Data Acquisition, Cleaning, and Preprocessing

Week 1 focused on acquiring the public dataset, understanding its structure, identifying data-quality problems, handling missing and inconsistent values, examining potential outliers, and creating useful derived features.

The main Week 1 script is:

```text
src/titanic_cleaning.py
```

The cleaned dataset is saved as:

```text
outputs/titanic_cleaned.csv
```

---

## Week 1 Workflow

```text
Kaggle Titanic Dataset
        │
        ▼
Load Dataset with Pandas
        │
        ▼
Initial Dataset Inspection
        │
        ├── First rows
        ├── Shape
        ├── Column names
        ├── Data types
        ├── Dataset information
        └── Descriptive statistics
        │
        ▼
Data Quality Assessment
        │
        ├── Missing values
        ├── Missing percentages
        ├── Duplicate rows
        ├── Duplicate PassengerIds
        ├── Categorical consistency
        └── Numerical validity checks
        │
        ▼
Outlier Analysis
        │
        ├── Age
        ├── Fare
        ├── SibSp
        └── Parch
        │
        ▼
Data Cleaning
        │
        ├── Age imputation
        ├── Embarked imputation
        ├── Cabin transformation
        └── Fare outlier treatment
        │
        ▼
Feature Engineering
        │
        ├── Cabin_known
        ├── Fare_capped
        ├── FamilySize
        └── IsAlone
        │
        ▼
Final Validation
        │
        ├── Missing-value check
        ├── Duplicate check
        └── Dataset-shape verification
        │
        ▼
Save Cleaned Dataset
        │
        ▼
outputs/titanic_cleaned.csv
```

---

## Initial Dataset Exploration

The original Titanic dataset was inspected using Pandas.

The analysis included:

- first five rows
- dataset dimensions
- column names
- data types
- complete dataset information
- descriptive statistics
- categorical values
- missing-value counts
- duplicate checks

The original dataset contained:

```text
891 rows
12 columns
```

---

## Missing-Value Analysis

Three original columns contained missing values:

| Feature | Missing Values | Percentage |
|---|---:|---:|
| `Age` | 177 | 19.87% |
| `Cabin` | 687 | 77.10% |
| `Embarked` | 2 | 0.22% |

A missing-value visualization was generated to make the scale of missing data easier to understand.

No other original columns contained missing values.

---

## Duplicate and Consistency Checks

The dataset was checked for:

- complete duplicate rows
- duplicate passenger identifiers
- unexpected categorical values
- invalid numerical values

Results:

- **Duplicate rows:** 0
- **Duplicate `PassengerId` values:** 0

Observed categorical values included:

```text
Sex:
male
female
```

```text
Embarked:
S
C
Q
```

```text
Pclass:
1
2
3
```

```text
Survived:
0
1
```

No unexpected categorical values were identified.

---

## Numerical Validation

The following numerical variables were checked for invalid negative values:

- `Age`
- `Fare`
- `SibSp`
- `Parch`

No invalid negative values were detected.

This helped distinguish genuine statistical outliers from erroneous records.

---

## Outlier Analysis

Potential outliers were identified using the **1.5 × IQR rule**.

The examined variables were:

- `Age`
- `Fare`
- `SibSp`
- `Parch`

The number of observations flagged was:

| Feature | Flagged Observations |
|---|---:|
| `Age` | 11 |
| `Fare` | 116 |
| `SibSp` | 46 |
| `Parch` | 213 |

Boxplots were generated to visualize these distributions.

The observations were not automatically removed because a statistical outlier does not necessarily represent incorrect data.

This was particularly important for `Parch`, where valid non-zero family relationships can be flagged because both the first and third quartiles are zero.

---

# Missing-Value Treatment

## Age

Missing `Age` values were filled using the median calculated within groups defined by:

- `Pclass`
- `Sex`

This was preferred over one overall median because passenger age distributions vary across class and sex groups.

A global median fallback was also included in case any missing values remained.

After imputation:

```text
Missing Age values = 0
```

---

## Embarked

The two missing `Embarked` values were filled using the mode of the column.

After treatment:

```text
Missing Embarked values = 0
```

---

## Cabin

`Cabin` contained:

```text
687 missing values
77.10% missing
```

Because most cabin values were unavailable, artificial cabin numbers were not generated.

Instead, a binary feature named `Cabin_known` was created:

```text
1 = Cabin information is available
0 = Cabin information is unavailable
```

Final counts:

```text
Cabin unknown = 687
Cabin known   = 204
```

The original `Cabin` column was retained for transparency.

---

# Fare Outlier Treatment

`Fare` contained several extreme values.

Using the IQR rule, the upper limit was:

```text
65.6344
```

Instead of deleting passengers with high fares, the original `Fare` column was preserved and a new feature called `Fare_capped` was created.

Original maximum fare:

```text
512.3292
```

Maximum capped fare:

```text
65.6344
```

This approach reduces the influence of extreme values while preserving the original observations.

A before-and-after boxplot was generated to visualize the effect.

---

# Week 1 Feature Engineering

Several additional variables were created.

## `Cabin_known`

Indicates whether cabin information is available.

```text
1 = Known
0 = Unknown
```

## `Fare_capped`

Contains fare values after limiting observations above the upper IQR threshold.

The original `Fare` column remains unchanged.

## `FamilySize`

Calculated as:

```text
FamilySize = SibSp + Parch + 1
```

The additional `1` represents the passenger.

Summary:

- Minimum family size: **1**
- Maximum family size: **11**
- Mean family size: approximately **1.90**

## `IsAlone`

A binary feature derived from `FamilySize`.

```text
1 = Passenger travelled alone
0 = Passenger travelled with family
```

Final distribution:

| Travel Status | Count |
|---|---:|
| Alone | 537 |
| With family | 354 |

---

# Week 1 Final Validation

After cleaning and feature engineering, the dataset was validated again.

## Dataset Dimensions

Original dataset:

```text
(891, 12)
```

Cleaned dataset:

```text
(891, 16)
```

## Final Results

- Rows removed: **0**
- Duplicate rows: **0**
- Missing values in analysis-ready features: **0**
- Original source columns retained where useful
- Four engineered features created

The cleaned dataset is saved as:

```text
outputs/titanic_cleaned.csv
```

---

# Week 1 Visualizations

Week 1 figures are stored in:

```text
outputs/figures/
```

The cleaning script generates:

```text
missing_values.png
age_boxplot.png
fare_boxplot.png
sibsp_boxplot.png
parch_boxplot.png
fare_before_after.png
survival_by_sex.png
survival_by_class.png
age_distribution.png
```

Some basic survival visualizations were included in the original Week 1 workflow, while the dedicated exploratory analysis was expanded substantially during Week 2.

---

# Week 1 Key Decisions

Important preprocessing decisions included:

- using grouped medians rather than one overall value for missing `Age`
- using the mode for the two missing `Embarked` entries
- avoiding artificial cabin-number imputation
- retaining statistical outliers when they represented plausible passenger data
- preserving the original `Fare` column
- creating a capped fare variable rather than deleting high-fare records
- retaining all 891 passengers
- creating family-related and cabin-related variables for later analysis

---

# Impact of Week 1 Preprocessing

The cleaning process improved the dataset's suitability for further analysis and modeling.

However, transformations also affect the statistical properties of the data.

For example:

- Age imputation replaces unknown ages with estimates.
- Fare capping reduces the magnitude of extreme fares.
- `Cabin_known` simplifies cabin information into an availability indicator.
- `FamilySize` combines two family-related variables into a more interpretable feature.

Original variables were retained where useful so preprocessing decisions remain transparent and auditable.

---

# Week 2 — Exploratory Data Analysis and Visualization

Week 2 builds directly on the cleaned dataset generated during Week 1.

The objective is to identify meaningful distributions, trends, associations, and anomalies using Pandas, Matplotlib, and Seaborn.

The Week 2 script is:

```text
src/titanic_eda.py
```

The input dataset is:

```text
outputs/titanic_cleaned.csv
```

Figures are stored in:

```text
outputs/eda_figures/
```

---

# Week 2 Workflow

```text
Cleaned Titanic Dataset
        │
        ▼
Dataset Overview
        │
        ├── Shape
        ├── Columns
        ├── Data types
        ├── Statistics
        ├── Missing values
        └── Duplicate check
        │
        ▼
Univariate Analysis
        │
        ├── Survival
        ├── Passenger class
        ├── Sex
        ├── Age
        ├── Fare
        ├── Embarkation
        ├── Family size
        ├── Travelling status
        └── Cabin availability
        │
        ▼
Bivariate Analysis
        │
        ├── Survival vs Sex
        ├── Survival vs Class
        ├── Survival vs Embarkation
        ├── Survival vs IsAlone
        ├── Survival vs Cabin_known
        ├── Survival vs Age
        ├── Survival vs Fare
        └── Survival vs FamilySize
        │
        ▼
Multivariate Analysis
        │
        ├── Sex + Class + Survival
        ├── Age + Fare + Class + Survival
        └── Class + Cabin + Survival
        │
        ▼
Correlation Analysis
        │
        ▼
Interpretation and Key Insights
```

---

# Week 2 Dataset Overview

The cleaned dataset contains:

```text
891 rows
16 columns
```

The EDA script confirms:

- dataset dimensions
- column names
- data types
- descriptive statistics
- missing values
- duplicate records

The original `Cabin` variable still contains missing values because it is intentionally preserved.

The engineered `Cabin_known` feature is used when analyzing cabin-information availability.

---

# Univariate Analysis

Features examined include:

- `Survived`
- `Pclass`
- `Sex`
- `Age`
- `Fare_capped`
- `Embarked`
- `FamilySize`
- `IsAlone`
- `Cabin_known`

---

## Survival Distribution

| Survival Status | Passengers | Percentage |
|---|---:|---:|
| Did not survive | 549 | 61.62% |
| Survived | 342 | 38.38% |

The dataset contains more non-survivors than survivors.

---

## Passenger Class Distribution

| Class | Passengers |
|---|---:|
| 1st Class | 216 |
| 2nd Class | 184 |
| 3rd Class | 491 |

Third-class passengers form the largest group.

---

## Sex Distribution

| Sex | Passengers |
|---|---:|
| Male | 577 |
| Female | 314 |

Male passengers represent the majority of the dataset.

---

## Age Distribution

After Week 1 preprocessing:

- Mean age: **29.11 years**
- Median age: **26 years**
- Minimum age: **0.42 years**
- Maximum age: **80 years**

Most passengers are concentrated among younger and middle-aged adults.

---

## Fare Distribution

Using `Fare_capped`:

- Mean: **24.05**
- Median: **14.45**
- Maximum: **65.6344**

The capped version prevents a small number of extreme fares from dominating the visualization.

---

## Embarkation Port

| Port | Passengers | Percentage |
|---|---:|---:|
| Southampton | 646 | 72.50% |
| Cherbourg | 168 | 18.86% |
| Queenstown | 77 | 8.64% |

Southampton was the most common embarkation port.

---

## Family Size

Family size is concentrated at smaller values.

The most common family size is:

```text
1
```

This represents passengers travelling alone.

---

## Travelling Alone

| Travel Status | Passengers | Percentage |
|---|---:|---:|
| With family | 354 | 39.73% |
| Alone | 537 | 60.27% |

Most passengers travelled alone.

---

## Cabin Information Availability

| Cabin Information | Passengers | Percentage |
|---|---:|---:|
| Unknown | 687 | 77.10% |
| Known | 204 | 22.90% |

Only around one-quarter of passengers had recorded cabin information.

---

# Bivariate Analysis

Bivariate analysis examines relationships between two variables, with survival used as the primary comparison variable.

---

## Survival by Sex

| Sex | Survival Rate |
|---|---:|
| Female | 74.20% |
| Male | 18.89% |

Female passengers had substantially higher observed survival.

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
|---|---:|
| 1st Class | 62.96% |
| 2nd Class | 47.28% |
| 3rd Class | 24.24% |

Observed survival decreased substantially from first class to third class.

---

## Survival by Embarkation Port

| Port | Survival Rate |
|---|---:|
| Cherbourg | 55.36% |
| Queenstown | 38.96% |
| Southampton | 33.90% |

Cherbourg passengers had the highest observed survival rate.

However, embarkation should not be interpreted as a direct cause because passenger composition differs between ports.

---

## Survival by Travelling Status

| Travel Status | Survival Rate |
|---|---:|
| With family | 50.56% |
| Alone | 30.35% |

Passengers travelling with family had higher observed survival.

---

## Survival by Cabin Information

| Cabin Information | Survival Rate |
|---|---:|
| Unknown | 29.99% |
| Known | 66.67% |

Passengers with known cabin information showed substantially higher observed survival.

Cabin availability is also strongly associated with passenger class.

---

## Age and Survival

| Survival Status | Mean Age | Median Age |
|---|---:|---:|
| Did not survive | 29.74 | 25 |
| Survived | 28.11 | 27 |

Age distributions overlap considerably.

Age therefore has only a weak overall relationship with survival when examined independently.

---

## Fare and Survival

| Survival Status | Mean Fare | Median Fare |
|---|---:|---:|
| Did not survive | 18.92 | 10.50 |
| Survived | 32.28 | 26.00 |

Surviving passengers generally paid higher fares.

Fare is also strongly related to passenger class.

---

## Survival by Family Size

| Family Size | Count | Survival Rate |
|---:|---:|---:|
| 1 | 537 | 30.35% |
| 2 | 161 | 55.28% |
| 3 | 102 | 57.84% |
| 4 | 29 | 72.41% |
| 5 | 15 | 20.00% |
| 6 | 22 | 13.64% |
| 7 | 12 | 33.33% |
| 8 | 6 | 0.00% |
| 11 | 7 | 0.00% |

Moderate family sizes show relatively high observed survival.

Very large family-size groups contain few passengers, so their percentages should be interpreted cautiously.

---

# Multivariate Analysis

Multivariate analysis examines several passenger characteristics simultaneously.

---

## Survival by Passenger Class and Sex

| Passenger Class | Sex | Count | Survival Rate |
|---:|---|---:|---:|
| 1 | Female | 94 | 96.81% |
| 1 | Male | 122 | 36.89% |
| 2 | Female | 76 | 92.11% |
| 2 | Male | 108 | 15.74% |
| 3 | Female | 144 | 50.00% |
| 3 | Male | 347 | 13.54% |

Female passengers show higher observed survival within every passenger class.

Passenger class also remains important within each sex group.

---

## Passenger Class and Cabin Availability

| Class | Cabin Known | Count | Survival Rate |
|---:|---:|---:|---:|
| 1 | No | 40 | 47.50% |
| 1 | Yes | 176 | 66.48% |
| 2 | No | 168 | 44.05% |
| 2 | Yes | 16 | 81.25% |
| 3 | No | 479 | 23.59% |
| 3 | Yes | 12 | 50.00% |

Known-cabin passengers show higher observed survival within each passenger class.

The second- and third-class known-cabin groups contain relatively few passengers, so these percentages should be interpreted cautiously.

---

## Age, Fare, Survival, and Passenger Class

A multivariate scatter plot examines:

- Age
- `Fare_capped`
- Survival status
- Passenger class

The visualization demonstrates substantial overlap between passenger groups while also showing the relationship between fare and passenger class.

---

# Correlation Analysis

Pearson correlation was calculated for numerical and engineered features.

## Correlation with Survival

| Feature | Correlation |
|---|---:|
| `Pclass` | -0.34 |
| `Fare_capped` | +0.32 |
| `Cabin_known` | +0.32 |
| `IsAlone` | -0.20 |
| `Parch` | +0.08 |
| `Age` | -0.06 |
| `SibSp` | -0.04 |
| `FamilySize` | +0.02 |

Passenger class, fare, and cabin-information availability show some of the strongest linear relationships with survival among the numerical variables.

## Relationships Between Predictors

Several stronger relationships also exist between predictor variables:

```text
Pclass vs Fare_capped       = -0.72
Pclass vs Cabin_known       = -0.73
Fare_capped vs Cabin_known  = +0.62
FamilySize vs SibSp         = +0.89
FamilySize vs Parch         = +0.78
FamilySize vs IsAlone       = -0.69
```

Some of these relationships are expected.

`FamilySize` is directly constructed from `SibSp` and `Parch`, so strong correlations with these variables are natural.

Passenger class, fare, and cabin availability also represent related socioeconomic and travel characteristics.

Correlation measures association and should not be interpreted as evidence of causation.

---

# Week 2 Visualizations

Week 2 figures are stored in:

```text
outputs/eda_figures/
```

The EDA script generates **22 visualizations**.

## Univariate Visualizations

- Survival distribution
- Passenger class distribution
- Sex distribution
- Age distribution
- Fare distribution
- Embarkation distribution
- Family-size distribution
- Travelling-alone distribution
- Cabin-information distribution

## Bivariate Visualizations

- Survival by sex
- Survival by passenger class
- Survival by embarkation port
- Survival by travelling status
- Survival by cabin availability
- Age by survival status
- Age distribution by survival status
- Fare by survival status
- Survival by family size

## Multivariate and Correlation Visualizations

- Survival by class and sex
- Age vs fare by survival status and class
- Survival by class and cabin availability
- Correlation heatmap

---

# Key Week 2 EDA Findings

The Week 2 analysis identified several important patterns:

- **38.38%** of passengers survived.
- Female passengers had substantially higher observed survival than male passengers.
- First-class passengers had much higher survival than third-class passengers.
- Survivors generally paid higher fares.
- Passengers travelling with family had higher survival than solo travellers.
- Moderate family sizes generally had better observed outcomes.
- Passengers with known cabin information showed higher observed survival.
- Age alone had only a weak linear relationship with survival.
- Fare, passenger class, and cabin availability were strongly related.
- Sex and passenger class together produced some of the clearest survival differences.

---

# Interpretation Considerations

The EDA identifies **associations**, not causal relationships.

Several variables are related to one another.

For example:

- Fare is strongly associated with passenger class.
- Cabin availability is associated with passenger class.
- Embarkation groups may contain different passenger-class and sex distributions.
- `FamilySize`, `SibSp`, and `Parch` are mathematically related.

Small subgroups should also be interpreted cautiously.

For example, family-size groups of 8 and 11 contain very few passengers, so their survival percentages are less stable than results based on larger groups.

---

# Week 4 — Supervised Learning Model Implementation

Week 4 extends the data-preprocessing work from Week 1 and the exploratory findings from Week 2 into a supervised binary-classification problem.

The objective is to predict whether a Titanic passenger survived using demographic, socioeconomic, and travel-related characteristics.

The target variable is:

```text
Survived

0 = Did not survive
1 = Survived
```

The main Week 4 script is:

```text
src/titanic_supervised_ml.py
```

Reusable preprocessing logic required by the serialized model is stored in:

```text
src/titanic_transformers.py
```

---

# Week 4 Modeling Workflow

```text
Raw Titanic Dataset
        │
        ▼
80/20 Stratified Train/Test Split
        │
        ├── Training Set: 712 passengers
        └── Test Set: 179 passengers
        │
        ▼
Leakage-Safe Week 1 Preprocessing
        │
        ▼
5-Fold Stratified Cross-Validation
        │
        ▼
Baseline Model Comparison
        │
        ├── Logistic Regression
        ├── Decision Tree
        └── Random Forest
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Final Model Selection
        │
        ▼
Random Forest
        │
        ▼
Independent Holdout Evaluation
        │
        ▼
Feature Importance
        │
        ▼
Error and Subgroup Analysis
        │
        ▼
Feature Engineering Evaluation
        │
        ▼
Probability Calibration
```

---

# Leakage-Safe Preprocessing

Week 4 follows the preprocessing strategies established in Week 1:

- Age imputation using medians grouped by `Pclass × Sex`
- Embarked mode imputation
- Fare outlier capping
- `Cabin_known`
- `FamilySize`
- `IsAlone`

However, supervised modeling introduces an important methodological improvement.

Statistics used for preprocessing are learned from the **training data only**.

This means information from validation folds or the final holdout test set is not used to calculate:

- Age-imputation medians
- Embarked mode
- Fare median
- Fare-capping threshold

The preprocessing and feature-engineering logic is implemented within the Scikit-learn pipeline to reduce the risk of data leakage.

---

# Train/Test Split

The dataset was divided using an **80/20 stratified split**.

| Dataset | Passengers |
|---|---:|
| Training set | 712 |
| Holdout test set | 179 |

Training target distribution:

```text
Did not survive ≈ 61.66%
Survived        ≈ 38.34%
```

Testing target distribution:

```text
Did not survive ≈ 61.45%
Survived        ≈ 38.55%
```

Stratification preserved approximately the same class distribution in both subsets.

The holdout test set was not used during model selection or hyperparameter tuning.

---

# Baseline Model Comparison

Three supervised classification algorithms were evaluated using **5-fold stratified cross-validation**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8062 | 0.7569 | 0.7288 | 0.7423 | **0.8681** |
| Logistic Regression | 0.8048 | **0.7637** | 0.7145 | 0.7371 | 0.8606 |
| Decision Tree | 0.7725 | 0.6962 | 0.7253 | 0.7101 | 0.7620 |

Random Forest achieved the strongest baseline ROC-AUC and F1 score.

Logistic Regression performed competitively, while the unrestricted Decision Tree produced weaker generalization performance.

---

# Hyperparameter Tuning

`GridSearchCV` was used to optimize the candidate models.

**ROC-AUC** was used as the primary model-selection metric.

## Selected Random Forest Parameters

```text
class_weight = balanced
max_depth = None
max_features = sqrt
min_samples_leaf = 5
n_estimators = 400
```

## Tuned Model Comparison

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0.8175** | 0.7596 | **0.7802** | **0.7675** | **0.8832** |
| Logistic Regression | 0.7992 | 0.7576 | 0.7034 | 0.7286 | 0.8641 |
| Decision Tree | 0.7949 | 0.7477 | 0.7288 | 0.7295 | 0.8564 |

Random Forest was selected as the final classifier.

---

# Effect of Hyperparameter Tuning

| Model | Baseline ROC-AUC | Tuned ROC-AUC | Improvement |
|---|---:|---:|---:|
| Random Forest | 0.8681 | 0.8832 | +0.0150 |
| Logistic Regression | 0.8606 | 0.8641 | +0.0035 |
| Decision Tree | 0.7620 | 0.8564 | +0.0943 |

Hyperparameter tuning improved all three models.

The Decision Tree showed the largest improvement because restricting its complexity substantially reduced the weakness of the unrestricted baseline tree.

---

# Final Holdout Performance

After model selection and tuning were completed using only training data, the selected Random Forest was evaluated on the independent **179-passenger holdout test set**.

| Metric | Result |
|---|---:|
| Accuracy | **0.7709** |
| Balanced Accuracy | **0.7704** |
| Precision | **0.6795** |
| Recall | **0.7681** |
| F1 Score | **0.7211** |
| ROC-AUC | **0.8366** |
| Average Precision | **0.8198** |
| Brier Score | **0.1525** |

The tuned cross-validation ROC-AUC was:

```text
0.8832
```

The independent holdout ROC-AUC was:

```text
0.8366
```

The difference indicates that the test sample was somewhat more challenging than the cross-validation folds, while the model still retained useful discriminatory ability on unseen passengers.

---

# Classification Report

The final test classification results were:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Did Not Survive | 0.8416 | 0.7727 | 0.8057 | 110 |
| Survived | 0.6795 | 0.7681 | 0.7211 | 69 |

Overall test accuracy:

```text
77.09%
```

The selected model achieved relatively strong survivor recall, detecting approximately **76.81%** of actual survivors in the holdout sample.

---

# Confusion Matrix

The final confusion matrix was:

```text
[[85 25]
 [16 53]]
```

This corresponds to:

| Prediction Result | Count |
|---|---:|
| True Negatives | 85 |
| False Positives | 25 |
| False Negatives | 16 |
| True Positives | 53 |

The model correctly classified:

```text
138 / 179 passengers
```

and produced:

```text
41 classification errors
```

---

# Feature Importance

Random Forest feature importance was analyzed after final model selection.

## Aggregated Feature Importance

| Feature | Importance |
|---|---:|
| `Sex` | **0.4481** |
| `Fare_capped` | 0.1502 |
| `Age` | 0.1162 |
| `Pclass` | 0.0869 |
| `Cabin_known` | 0.0689 |
| `FamilySize` | 0.0467 |
| `Embarked` | 0.0346 |
| `SibSp` | 0.0190 |
| `Parch` | 0.0164 |
| `IsAlone` | 0.0132 |

`Sex` was the strongest predictive feature in the final Random Forest.

This aligns with Week 2 EDA, where passenger sex showed one of the clearest observed relationships with survival.

Fare, age, passenger class, and cabin-information availability also contributed meaningfully.

Random Forest feature importance represents **predictive contribution**, not causal influence.

---

# Feature Engineering Effectiveness

A controlled experiment evaluated whether the Week 1 engineered feature set improved model performance.

The same Random Forest configuration and cross-validation approach were used for both feature sets.

| Feature Set | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|
| Basic Features | 0.8077 | 0.7534 | 0.8791 |
| Week 1 Engineered Features | **0.8175** | **0.7675** | **0.8832** |

ROC-AUC improvement from Week 1 feature engineering:

```text
+0.0040
```

The improvement is modest but measurable.

The engineered feature set also improved accuracy and F1 score.

---

# Error and Subgroup Analysis

The final model's errors were analyzed instead of relying only on aggregate metrics.

## Prediction-Type Distribution

```text
True Negative     85
True Positive     53
False Positive    25
False Negative    16
```

## Performance by Sex

| Sex | Passengers | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Female | 61 | 0.8361 | 0.8182 | 1.0000 | 0.9000 | 0.8444 |
| Male | 118 | 0.7373 | 0.3478 | 0.3333 | 0.3404 | 0.6443 |

The model performed noticeably differently across male and female passengers in this test split.

In particular, survivor recall was much higher among female passengers.

These results describe this specific holdout sample and should not be generalized without caution.

---

## Performance by Passenger Class

| Class | Passengers | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 1st Class | 45 | 0.5778 | 0.8400 | 0.6885 | 0.7930 |
| 2nd Class | 34 | 0.8824 | 0.8500 | 0.8947 | 0.8893 |
| 3rd Class | 100 | 0.8200 | 0.6250 | 0.6250 | 0.7445 |

The variation demonstrates why one overall accuracy score cannot completely describe model behavior.

---

## Age-Group Considerations

Some age groups contained very few passengers.

For example, the senior subgroup contained only:

```text
6 passengers
```

Therefore, metrics for very small groups are unstable and should be interpreted cautiously.

---

# Most Confident Incorrect Predictions

The analysis also identified incorrect predictions made with relatively high confidence.

These examples demonstrate an important limitation of predictive models:

> A high model probability does not guarantee that an individual prediction is correct.

The detailed results are stored in:

```text
outputs/model_metrics/most_confident_errors.csv
```

---

# Probability Calibration

Probability calibration was evaluated after final model selection.

The model achieved:

```text
Brier Score = 0.1525
```

Lower Brier scores indicate more accurate probability predictions.

A calibration curve was also generated to compare:

```text
Mean Predicted Survival Probability
```

against:

```text
Observed Survival Frequency
```

The highest predicted-probability groups showed relatively close agreement between predicted and observed survival frequencies, while some lower and middle probability ranges showed greater variation.

Calibration was used only as a post-model diagnostic and was **not used for additional hyperparameter tuning**.

---

# Week 4 Visualizations

Week 4 figures are stored in:

```text
outputs/model_figures/
```

The supervised-learning workflow generates:

```text
01_final_confusion_matrix.png
02_final_roc_curve.png
03_final_precision_recall_curve.png
04_feature_importance.png
05_subgroup_error_rates.png
06_feature_engineering_comparison.png
07_baseline_vs_tuned_models.png
08_calibration_curve.png
```

These figures provide visual evidence for:

- classification performance
- discrimination ability
- precision-recall behavior
- feature contribution
- subgroup error rates
- feature-engineering effectiveness
- hyperparameter-tuning improvements
- probability calibration

---

# Week 4 Model Metrics and Diagnostics

Detailed Week 4 results are stored in:

```text
outputs/model_metrics/
```

Files include:

```text
baseline_model_comparison.csv
tuned_model_comparison.csv
best_hyperparameters.json
final_test_metrics.json
classification_report.txt
test_predictions.csv
detailed_feature_importance.csv
aggregated_feature_importance.csv
subgroup_performance.csv
test_error_analysis.csv
most_confident_errors.csv
feature_engineering_comparison.csv
baseline_vs_tuned.csv
calibration_data.csv
```

---

# Saved Machine Learning Pipeline

The complete trained pipeline is stored as:

```text
models/titanic_random_forest_pipeline.joblib
```

The serialized pipeline contains:

```text
Raw passenger features
        ↓
Week 1 feature engineering
        ↓
Numerical preprocessing
        ↓
Categorical encoding
        ↓
Tuned Random Forest classifier
        ↓
Survival prediction
```

The custom transformer is defined in:

```text
src/titanic_transformers.py
```

This allows the saved Joblib model to be loaded in a fresh Python process.

Example validation:

```python
import sys
import joblib

sys.path.insert(0, "src")

model = joblib.load("models/titanic_random_forest_pipeline.joblib")

print(type(model))
```

Expected result:

```text
<class 'sklearn.pipeline.Pipeline'>
```

---

# Technologies Used

| Week 1 | Week 2 | Week 4 |
|--------|--------|--------|
| Python | Python | Python |
| Pandas | Pandas | Pandas |
| Matplotlib | Matplotlib | Matplotlib |
| Seaborn | Seaborn | NumPy |
| pathlib | pathlib | pathlib|
| uv | uv | uv|
| Visual Studio Code | Visual Studio Code | Visual Studio Code |
| Git | Git | Git |
| GitHub | GitHub | GitHub |
|  |  | Scikit-learn |
|  |  | Joblib |

---

# Reports

Final internship reports are stored in:

```text
reports/
├── Week_1_Data_Cleaning_and_Preprocessing_Report.docx
├── Week_2_Titanic_EDA_Report.docx
└── Week_4_Supervised_Learning_Report.docx
```

Each internship week has a separate report even though Weeks 1, 2, and 4 form one progressively developed Titanic project.

---

# Project Structure

```text
Titanic_Data_Science_Project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── titanic_cleaning.py
│   ├── titanic_eda.py
│   ├── titanic_supervised_ml.py
│   └── titanic_transformers.py
│
├── outputs/
│   ├── titanic_cleaned.csv
│   │
│   ├── figures/
│   │   └── Week 1 cleaning and preprocessing figures
│   │
│   ├── eda_figures/
│   │   └── Week 2 exploratory-analysis figures
│   │
│   ├── model_figures/
│   │   └── Week 4 model-evaluation figures
│   │
│   └── model_metrics/
│       └── Week 4 metrics and diagnostic outputs
│
├── models/
│   └── titanic_random_forest_pipeline.joblib
│
├── reports/
│   ├── Week_1_Data_Cleaning_and_Preprocessing_Report.docx
│   ├── Week_2_Titanic_EDA_Report.docx
│   └── Week_4_Supervised_Learning_Report.docx
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/ARraj14/Titanic_Data_Science_Project.git
cd Titanic_Data_Science_Project
```

## 2. Install Dependencies

Using `uv`:

```bash
uv sync
```

---

## 3. Run Week 1 — Data Cleaning

```bash
uv run python src/titanic_cleaning.py
```

This generates:

```text
outputs/titanic_cleaned.csv
```

and Week 1 figures in:

```text
outputs/figures/
```

---

## 4. Run Week 2 — Exploratory Data Analysis

After generating the cleaned dataset:

```bash
uv run python src/titanic_eda.py
```

Week 2 figures are saved in:

```text
outputs/eda_figures/
```

---

## 5. Run Week 4 — Supervised Learning

```bash
uv run python src/titanic_supervised_ml.py
```

Week 4 figures are saved in:

```text
outputs/model_figures/
```

Detailed metrics and diagnostic files are saved in:

```text
outputs/model_metrics/
```

The final trained pipeline is saved in:

```text
models/titanic_random_forest_pipeline.joblib
```

---

# Challenges and Solutions

## Missing Age Values

Instead of using a single overall median, missing `Age` values were filled using medians calculated within passenger-class and sex groups.

This retains more group-specific information.

---

## Missing Cabin Information

Because most `Cabin` values were unavailable, cabin numbers were not artificially imputed.

The `Cabin_known` indicator instead preserves information about whether a cabin record exists.

---

## Statistical Outliers

The IQR rule was used to identify unusual observations, but values were interpreted in context rather than automatically deleted.

Statistical outliers can still represent valid passengers.

---

## Extreme Fare Values

The original `Fare` variable was preserved.

A separate capped variable was created so that extreme fares would have less influence while no passenger records were removed.

---

## Preserving Passenger Records

No passenger rows were removed during Week 1 data cleaning.

The cleaned dataset retains all **891 passengers**.

---

## Interpreting EDA Results

Relationships discovered during EDA were interpreted as associations rather than proof of causation.

Sample size was also considered when interpreting small passenger subgroups.

---

## Preventing Data Leakage

The Week 1 cleaned dataset was appropriate for EDA because preprocessing was performed on the complete dataset.

For Week 4 supervised modeling, however, preprocessing was reproduced inside the Scikit-learn pipeline.

Learned statistics were therefore calculated from training data only.

This prevents validation and holdout-test information from leaking into model training.

---

## Model Selection

Three classifiers were evaluated using the same stratified cross-validation strategy.

The final model was selected using cross-validated **ROC-AUC** rather than holdout-test performance.

This keeps the final test set independent from model selection.

---

## Controlling Overfitting

Training and validation ROC-AUC values were compared during hyperparameter tuning.

For the selected Random Forest:

```text
Training ROC-AUC = 0.9379
CV ROC-AUC       = 0.8832
Gap              = 0.0547
```

A minimum leaf size of five and repeated cross-validation evaluation helped reduce excessive fitting to individual training records.

---

## Model Portability

The custom Titanic feature transformer was placed in:

```text
src/titanic_transformers.py
```

instead of being defined only inside the training script.

This allows the serialized Joblib pipeline to be imported and loaded correctly in a fresh Python process.

---

# Key Learnings

This project provided practical experience with:

- public dataset acquisition
- dataset exploration
- data-quality assessment
- missing-value analysis
- grouped imputation
- duplicate detection
- categorical consistency checking
- numerical validation
- IQR-based outlier detection
- contextual interpretation of outliers
- feature engineering
- data transformation
- data validation
- univariate analysis
- bivariate analysis
- multivariate analysis
- Pandas aggregation
- correlation analysis
- Matplotlib visualization
- Seaborn visualization
- interpretation of statistical relationships
- supervised binary classification
- stratified train/test splitting
- leakage-safe preprocessing
- Scikit-learn pipelines
- Logistic Regression
- Decision Trees
- Random Forests
- 5-fold stratified cross-validation
- GridSearchCV hyperparameter tuning
- Accuracy evaluation
- Balanced Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Average Precision
- confusion-matrix interpretation
- ROC curves
- precision-recall curves
- feature importance
- controlled feature-engineering comparison
- model error analysis
- subgroup performance analysis
- probability calibration
- Brier score analysis
- model serialization with Joblib
- reproducible project organization
- dependency management with `uv`
- Git version control
- GitHub repository management

---

# Conclusion

The **Titanic Data Science Project** demonstrates a progressive end-to-end workflow covering data preparation, exploratory analysis, and supervised machine learning.

During **Week 1**, the original Titanic dataset was inspected, cleaned, validated, and transformed. Missing values were handled using variable-specific strategies, potential outliers were examined rather than blindly removed, and engineered features such as `Cabin_known`, `Fare_capped`, `FamilySize`, and `IsAlone` were created.

The cleaned dataset retained all **891 passenger records** and expanded from **12 original columns to 16 columns**.

During **Week 2**, the cleaned dataset was analyzed using univariate, bivariate, multivariate, and correlation-based techniques. A total of **22 visualizations** were generated to examine distributions and relationships.

The EDA identified important associations between survival and passenger sex, passenger class, fare, travelling status, family characteristics, and cabin-information availability.

During **Week 4**, these findings were extended into a supervised binary-classification problem.

Logistic Regression, Decision Tree, and Random Forest classifiers were evaluated using **5-fold stratified cross-validation** and subsequently tuned using `GridSearchCV`.

The tuned Random Forest achieved a cross-validated ROC-AUC of:

```text
0.8832
```

and was selected without using holdout-test results.

On the independent 179-passenger test set, it achieved:

```text
Accuracy = 77.09%
Recall   = 76.81%
F1 Score = 72.11%
ROC-AUC  = 0.8366
```

Additional analysis included:

- feature importance
- subgroup performance
- detailed error analysis
- feature-engineering effectiveness
- baseline-versus-tuned model comparison
- probability calibration

The Week 1 engineered feature set produced a modest but measurable improvement in cross-validated ROC-AUC from **0.8791 to 0.8832**.

Together, Weeks 1, 2, and 4 demonstrate how the same public dataset can progress from raw-data preparation through detailed exploratory analysis and into a reproducible machine-learning pipeline.

---

# Dataset Source

**Titanic - Machine Learning from Disaster**  
Kaggle

---

# Project Repository

```text
https://github.com/ARraj14/Titanic_Data_Science_Project
```