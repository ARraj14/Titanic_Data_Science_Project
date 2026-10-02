import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Reproduces the Week 1 Titanic preprocessing and feature engineering.

    All learned statistics are calculated from the data passed to fit(),
    which prevents data leakage during cross-validation and testing.
    """

    def fit(self, X, y=None):
        X = X.copy()

        self.age_group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()

        self.age_global_median_ = X["Age"].median()

        self.embarked_mode_ = X["Embarked"].mode()[0]

        q1 = X["Fare"].quantile(0.25)
        q3 = X["Fare"].quantile(0.75)

        iqr = q3 - q1

        self.fare_upper_cap_ = q3 + (1.5 * iqr)
        self.fare_median_ = X["Fare"].median()

        return self

    def transform(self, X):
        X = X.copy()

        # Age imputation
        def fill_age(row):
            if pd.notna(row["Age"]):
                return row["Age"]

            group_key = (
                row["Pclass"],
                row["Sex"],
            )

            return self.age_group_medians_.get(
                group_key,
                self.age_global_median_,
            )

        X["Age"] = X.apply(
            fill_age,
            axis=1,
        )

        # Embarked imputation
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)

        # Fare handling
        X["Fare"] = X["Fare"].fillna(self.fare_median_)

        X["Fare_capped"] = X["Fare"].clip(upper=self.fare_upper_cap_)

        # Cabin information
        X["Cabin_known"] = X["Cabin"].notna().astype(int)

        # Family features
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        final_features = [
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

        return X[final_features]
