# Titanic Dataset Cleaning and Preprocessing Project

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths

DATA_PATH = Path("data/train.csv")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset

df = pd.read_csv(DATA_PATH)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

#  Missing-value analysis

print("\nMissing value analysis:")

missing_report = pd.DataFrame(
    {
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (df.isnull().sum() / len(df) * 100).round(2),
    }
)

missing_report = missing_report.sort_values(by="Missing Count", ascending=False)

print(missing_report)

# Missing-value visualization

missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]

plt.figure(figsize=(8, 5))

sns.barplot(x=missing_counts.values, y=missing_counts.index)

plt.title("Missing Values by Column")
plt.xlabel("Number of Missing Values")
plt.ylabel("Column")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

# Duplicate and consistency checks

print("\nDuplicate row analysis:")

duplicate_rows = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_rows)

print("\nDuplicate PassengerId values:")
print(df["PassengerId"].duplicated().sum())

print("\nUnique values in categorical columns:")

print("\nSex:")
print(df["Sex"].unique())

print("\nEmbarked:")
print(df["Embarked"].unique())

print("\nPclass:")
print(df["Pclass"].unique())

print("\nSurvived:")
print(df["Survived"].unique())

# Numerical value validation

print("\nNumerical value validation:")

print("Negative Age values:", (df["Age"] < 0).sum())
print("Negative Fare values:", (df["Fare"] < 0).sum())
print("Negative SibSp values:", (df["SibSp"] < 0).sum())
print("Negative Parch values:", (df["Parch"] < 0).sum())

# Outlier analysis using IQR

print("\nOutlier analysis using IQR:")

numerical_columns = ["Age", "Fare", "SibSp", "Parch"]

for column in numerical_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_condition = (df[column] < lower_bound) | (df[column] > upper_bound)

    outlier_count = outlier_condition.sum()

    print(f"\n{column}:")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {outlier_count}")

# Outlier visualization

print("\nCreating boxplots...")

for column in numerical_columns:
    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{column.lower()}_boxplot.png", dpi=300, bbox_inches="tight"
    )

    plt.show()
    plt.close()

# Data cleaning

cleaned_df = df.copy()

# ---- Age ----
# Fill missing Age values using median
# grouped by passenger class and sex.

cleaned_df["Age"] = cleaned_df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Global fallback in case any missing value remains.
cleaned_df["Age"] = cleaned_df["Age"].fillna(cleaned_df["Age"].median())

# ---- Embarked ----
# Fill missing categorical values using the mode.

cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(cleaned_df["Embarked"].mode()[0])

# ---- Cabin ----
# Create an indicator showing whether cabin
# information is available.

cleaned_df["Cabin_known"] = cleaned_df["Cabin"].notna().astype(int)

# Fare outlier treatment

q1 = cleaned_df["Fare"].quantile(0.25)
q3 = cleaned_df["Fare"].quantile(0.75)

iqr = q3 - q1

upper_fare_limit = q3 + (1.5 * iqr)

print("\nFare outlier limit:")
print("Upper limit:", upper_fare_limit)

cleaned_df["Fare_capped"] = cleaned_df["Fare"].clip(upper=upper_fare_limit)

print("\nFare comparison:")
print(cleaned_df[["Fare", "Fare_capped"]].describe())

# Fare before/after visualization

plt.figure(figsize=(8, 5))

sns.boxplot(data=cleaned_df[["Fare", "Fare_capped"]])

plt.title("Fare Before and After Outlier Capping")
plt.ylabel("Fare")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "fare_before_after.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

# Feature engineering

cleaned_df["FamilySize"] = cleaned_df["SibSp"] + cleaned_df["Parch"] + 1

print("\nFamilySize Summary:")
print(cleaned_df["FamilySize"].describe())


cleaned_df["IsAlone"] = (cleaned_df["FamilySize"] == 1).astype(int)

print("\nIsAlone value counts:")
print(cleaned_df["IsAlone"].value_counts())

# Exploratory visualizations

# ---- Survival by sex ----

survival_by_sex = df.groupby("Sex")["Survived"].mean().reset_index()

survival_by_sex["Survived"] *= 100

plt.figure(figsize=(7, 5))

sns.barplot(data=survival_by_sex, x="Sex", y="Survived")

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "survival_by_sex.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

# ---- Survival by passenger class ----

survival_by_class = df.groupby("Pclass")["Survived"].mean().reset_index()

survival_by_class["Survived"] *= 100

plt.figure(figsize=(7, 5))

sns.barplot(data=survival_by_class, x="Pclass", y="Survived")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "survival_by_class.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

# ---- Age distribution ----

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Age", bins=30, kde=True)

plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "age_distribution.png", dpi=300, bbox_inches="tight")

plt.show()
plt.close()

# Final validation

print("\nFinal dataset validation:")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nCleaned dataset shape:")
print(cleaned_df.shape)


cleaned_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare_capped",
    "Cabin_known",
    "FamilySize",
    "IsAlone",
    "Sex",
    "Embarked",
]


print("\nMissing values in cleaned features:")
print(cleaned_df[cleaned_features].isnull().sum())


print("\nDuplicate rows in cleaned dataset:")
print(cleaned_df.duplicated().sum())


print("\nFinal cleaned feature columns:")
print(cleaned_features)


# Save cleaned dataset

cleaned_df.to_csv(
    OUTPUT_DIR / "titanic_cleaned.csv",
    index=False,
)

print("\nCleaned dataset saved successfully.")
print("File: outputs/titanic_cleaned.csv")
