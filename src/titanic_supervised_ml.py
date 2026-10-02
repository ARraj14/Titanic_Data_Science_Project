# Titanic Dataset - Week 4 Supervised Learning
# Survival Prediction using Classification Models

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from sklearn.metrics import brier_score_loss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from titanic_transformers import TitanicFeatureEngineer


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

CLEANED_DATA_PATH = PROJECT_ROOT / "outputs" / "titanic_cleaned.csv"

MODEL_FIGURES_DIR = PROJECT_ROOT / "outputs" / "model_figures"

MODEL_METRICS_DIR = PROJECT_ROOT / "outputs" / "model_metrics"

MODELS_DIR = PROJECT_ROOT / "models"


MODEL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# WEEK 1 PREPROCESSING FOR MACHINE LEARNING
# ---------------------------------------------------------

# ---------------------------------------------------------
# LOAD WEEK 1 DATA
# ---------------------------------------------------------

raw_df = pd.read_csv(RAW_DATA_PATH)

cleaned_df = pd.read_csv(CLEANED_DATA_PATH)


# ---------------------------------------------------------
# WEEK 4 DATA FOUNDATION
# ---------------------------------------------------------

print("=" * 70)
print("WEEK 4 - SUPERVISED LEARNING MODEL IMPLEMENTATION")
print("TITANIC SURVIVAL CLASSIFICATION")
print("=" * 70)


print("\n1. ORIGINAL TITANIC DATASET")

print("\nShape:")
print(raw_df.shape)

print("\nColumns:")
print(raw_df.columns.tolist())


print("\n2. WEEK 1 CLEANED DATASET")

print("\nShape:")
print(cleaned_df.shape)

print("\nColumns:")
print(cleaned_df.columns.tolist())


# ---------------------------------------------------------
# VERIFY WEEK 1 FEATURE ENGINEERING
# ---------------------------------------------------------

week1_engineered_features = [
    "Cabin_known",
    "Fare_capped",
    "FamilySize",
    "IsAlone",
]

print("\n3. WEEK 1 ENGINEERED FEATURES")

for feature in week1_engineered_features:
    if feature in cleaned_df.columns:
        print(f"{feature}: FOUND")
    else:
        print(f"{feature}: MISSING")


# ---------------------------------------------------------
# TARGET VARIABLE
# ---------------------------------------------------------

print("\n4. TARGET VARIABLE")

print("\nTarget:")
print("Survived")

print("\nTarget classes:")
print("0 = Did Not Survive")
print("1 = Survived")

print("\nTarget distribution:")

target_counts = cleaned_df["Survived"].value_counts().sort_index()

print(target_counts)

print("\nTarget distribution (%):")

target_percentages = (
    cleaned_df["Survived"].value_counts(normalize=True).sort_index().mul(100).round(2)
)

print(target_percentages)


# ---------------------------------------------------------
# WEEK 4 CANDIDATE FEATURES
# ---------------------------------------------------------

features = [
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

print("\n5. WEEK 4 CANDIDATE FEATURES")

for feature in features:
    print(feature)


# ---------------------------------------------------------
# DATA QUALITY CHECK
# ---------------------------------------------------------

print("\n6. DATA QUALITY BEFORE MODELING")

print("\nMissing values in candidate features:")

print(cleaned_df[features].isnull().sum())

print("\nMissing values in target:")

print(cleaned_df["Survived"].isnull().sum())

print("\nDuplicate rows:")

print(cleaned_df.duplicated().sum())


# ---------------------------------------------------------
# MODELING DATA
# ---------------------------------------------------------

X = cleaned_df[features].copy()
y = cleaned_df["Survived"].copy()


print("\n7. MODELING DATA")

print("\nFeature matrix X shape:")
print(X.shape)

print("\nTarget vector y shape:")
print(y.shape)


print("\n" + "=" * 70)
print("WEEK 4 DATA FOUNDATION VALIDATION COMPLETE")
print("=" * 70)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

print("\n")
print("=" * 70)
print("8. TRAIN / TEST SPLIT")
print("=" * 70)


# We use the RAW dataset here so that preprocessing statistics
# are learned from training data only.

X_raw = raw_df.drop(columns=["Survived"]).copy()
y = raw_df["Survived"].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)


print("\nTraining target distribution (%):")
print(y_train.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nTesting target distribution (%):")
print(y_test.value_counts(normalize=True).sort_index().mul(100).round(2))


# =========================================================
# COLUMN PREPROCESSING
# =========================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare_capped",
    "Cabin_known",
    "FamilySize",
    "IsAlone",
]

categorical_features = [
    "Sex",
    "Embarked",
]


column_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)


# =========================================================
# CLASSIFICATION MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}


# =========================================================
# CROSS-VALIDATION
# =========================================================

print("\n")
print("=" * 70)
print("9. 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


cv_results = []


for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            (
                "week1_feature_engineering",
                TitanicFeatureEngineer(),
            ),
            (
                "column_preprocessing",
                column_preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    result = {
        "Model": model_name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "Precision Mean": scores["test_precision"].mean(),
        "Recall Mean": scores["test_recall"].mean(),
        "F1 Mean": scores["test_f1"].mean(),
        "ROC-AUC Mean": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
    }

    cv_results.append(result)


# =========================================================
# MODEL COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame(cv_results)

comparison_df = comparison_df.sort_values(
    "ROC-AUC Mean",
    ascending=False,
).reset_index(drop=True)


print("\nCross-validation model comparison:\n")

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# Save results

comparison_path = MODEL_METRICS_DIR / "baseline_model_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False,
)


print("\nSaved model comparison to:")
print(comparison_path)


# =========================================================
# BEST BASELINE MODEL
# =========================================================

best_model_name = comparison_df.loc[0, "Model"]
best_auc = comparison_df.loc[0, "ROC-AUC Mean"]


print("\n")
print("=" * 70)
print("10. BASELINE MODEL SELECTION")
print("=" * 70)

print(f"\nBest baseline model: {best_model_name}")
print(f"Mean CV ROC-AUC: {best_auc:.4f}")


print("\nIMPORTANT:")
print("The final test set has NOT been used for model selection.")

print(
    "Model selection is based only on cross-validation "
    "performance from the training data."
)

print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON COMPLETE")
print("=" * 70)

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

print("\n")
print("=" * 70)
print("11. HYPERPARAMETER TUNING")
print("=" * 70)

print("\nHyperparameters are tuned using only the training data.")

print("ROC-AUC is used as the primary optimization metric.")

print("The final test set remains completely untouched.")


# ---------------------------------------------------------
# PARAMETER GRIDS
# ---------------------------------------------------------

parameter_grids = {
    "Logistic Regression": {
        "model__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    },
    "Decision Tree": {
        "model__max_depth": [
            3,
            5,
            7,
            None,
        ],
        "model__min_samples_leaf": [
            1,
            3,
            5,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    },
    "Random Forest": {
        "model__n_estimators": [
            200,
            400,
        ],
        "model__max_depth": [
            None,
            6,
        ],
        "model__min_samples_leaf": [
            1,
            3,
            5,
        ],
        "model__max_features": [
            "sqrt",
            0.7,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    },
}


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

tuning_results = []

best_estimators = {}

best_parameters = {}


# ---------------------------------------------------------
# GRID SEARCH
# ---------------------------------------------------------

for model_name, model in models.items():
    print("\n" + "-" * 70)
    print(f"Tuning: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "week1_feature_engineering",
                TitanicFeatureEngineer(),
            ),
            (
                "column_preprocessing",
                column_preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grids[model_name],
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------
    # BEST CONFIGURATION
    # ---------------------------------------------

    best_index = grid_search.best_index_

    results = grid_search.cv_results_

    best_cv_auc = results["mean_test_roc_auc"][best_index]

    best_cv_auc_std = results["std_test_roc_auc"][best_index]

    best_train_auc = results["mean_train_roc_auc"][best_index]

    best_accuracy = results["mean_test_accuracy"][best_index]

    best_precision = results["mean_test_precision"][best_index]

    best_recall = results["mean_test_recall"][best_index]

    best_f1 = results["mean_test_f1"][best_index]

    overfit_gap = best_train_auc - best_cv_auc

    tuning_results.append(
        {
            "Model": model_name,
            "CV Accuracy": best_accuracy,
            "CV Precision": best_precision,
            "CV Recall": best_recall,
            "CV F1": best_f1,
            "CV ROC-AUC": best_cv_auc,
            "ROC-AUC Std": best_cv_auc_std,
            "Train ROC-AUC": best_train_auc,
            "Train-CV AUC Gap": overfit_gap,
        }
    )

    best_estimators[model_name] = grid_search.best_estimator_

    best_parameters[model_name] = grid_search.best_params_

    print("\nBest parameters:")

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(f"\nBest mean CV ROC-AUC: {best_cv_auc:.4f}")

    print(f"ROC-AUC standard deviation: {best_cv_auc_std:.4f}")

    print(f"Mean training ROC-AUC: {best_train_auc:.4f}")

    print(f"Train-CV ROC-AUC gap: {overfit_gap:.4f}")


# =========================================================
# TUNED MODEL COMPARISON
# =========================================================

tuned_comparison_df = pd.DataFrame(tuning_results)

tuned_comparison_df = tuned_comparison_df.sort_values(
    "CV ROC-AUC",
    ascending=False,
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("12. TUNED MODEL COMPARISON")
print("=" * 70)

print(
    "\n"
    + tuned_comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------
# SAVE TUNING RESULTS
# ---------------------------------------------------------

tuned_results_path = MODEL_METRICS_DIR / "tuned_model_comparison.csv"

tuned_comparison_df.to_csv(
    tuned_results_path,
    index=False,
)


parameters_path = MODEL_METRICS_DIR / "best_hyperparameters.json"

with open(
    parameters_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        best_parameters,
        file,
        indent=4,
    )


print("\nSaved tuned comparison to:")
print(tuned_results_path)

print("\nSaved best hyperparameters to:")
print(parameters_path)


# =========================================================
# FINAL MODEL SELECTION FROM TRAINING DATA
# =========================================================

selected_model_name = tuned_comparison_df.iloc[0]["Model"]

selected_cv_auc = tuned_comparison_df.iloc[0]["CV ROC-AUC"]

selected_estimator = best_estimators[selected_model_name]


print("\n")
print("=" * 70)
print("13. FINAL MODEL SELECTION")
print("=" * 70)

print(f"\nSelected model: {selected_model_name}")

print(f"Cross-validated ROC-AUC: {selected_cv_auc:.4f}")


print("\nSelected hyperparameters:")

for parameter, value in best_parameters[selected_model_name].items():
    print(f"{parameter}: {value}")


print("\nIMPORTANT:")

print("The final model was selected without using the test-set results.")

print(
    "The 179-passenger test set remains "
    "unseen and will be evaluated only once "
    "after model selection."
)


print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 70)

# =========================================================
# FINAL TEST SET EVALUATION
# =========================================================

print("\n")
print("=" * 70)
print("14. FINAL TEST SET EVALUATION")
print("=" * 70)

print("\nThe selected model is now evaluated on the 179-passenger holdout test set.")

print("This is the first time the test set is used during model development.")


# ---------------------------------------------------------
# TEST PREDICTIONS
# ---------------------------------------------------------

y_pred = selected_estimator.predict(X_test)

y_prob = selected_estimator.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# FINAL PERFORMANCE METRICS
# ---------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    y_pred,
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

test_precision = precision_score(
    y_test,
    y_pred,
)

test_recall = recall_score(
    y_test,
    y_pred,
)

test_f1 = f1_score(
    y_test,
    y_pred,
)

test_roc_auc = roc_auc_score(
    y_test,
    y_prob,
)

test_average_precision = average_precision_score(
    y_test,
    y_prob,
)


print("\nFinal holdout test metrics:\n")

print(f"Accuracy:          {test_accuracy:.4f}")

print(f"Balanced Accuracy: {test_balanced_accuracy:.4f}")

print(f"Precision:         {test_precision:.4f}")

print(f"Recall:            {test_recall:.4f}")

print(f"F1 Score:          {test_f1:.4f}")

print(f"ROC-AUC:           {test_roc_auc:.4f}")

print(f"Average Precision: {test_average_precision:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n")
print("=" * 70)
print("15. CLASSIFICATION REPORT")
print("=" * 70)

report_text = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Did Not Survive",
        "Survived",
    ],
    digits=4,
)

print("\n")
print(report_text)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("=" * 70)
print("16. CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nConfusion matrix:")
print(cm)


tn, fp, fn, tp = cm.ravel()

print("\nDetailed confusion matrix counts:")

print(f"True Negatives  (correctly predicted non-survivors): {tn}")

print(f"False Positives (predicted survivor, actually not):  {fp}")

print(f"False Negatives (missed actual survivors):           {fn}")

print(f"True Positives  (correctly predicted survivors):     {tp}")


# ---------------------------------------------------------
# CONFUSION MATRIX FIGURE
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Did Not Survive",
        "Survived",
    ],
)

display.plot(
    ax=ax,
    values_format="d",
)

ax.set_title("Random Forest - Final Test Confusion Matrix")

plt.tight_layout()

confusion_matrix_path = MODEL_FIGURES_DIR / "01_final_confusion_matrix.png"

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# ROC CURVE
# =========================================================

print("\n")
print("=" * 70)
print("17. ROC CURVE")
print("=" * 70)


fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    y_prob,
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {test_roc_auc:.3f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Final Random Forest Model")

plt.legend()

plt.tight_layout()

roc_curve_path = MODEL_FIGURES_DIR / "02_final_roc_curve.png"

plt.savefig(
    roc_curve_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nFinal ROC-AUC: {test_roc_auc:.4f}")


# =========================================================
# PRECISION-RECALL CURVE
# =========================================================

print("\n")
print("=" * 70)
print("18. PRECISION-RECALL CURVE")
print("=" * 70)


precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_test,
    y_prob,
)


plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    label=(f"Random Forest (AP = {test_average_precision:.3f})"),
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve - Final Random Forest Model")

plt.legend()

plt.tight_layout()

pr_curve_path = MODEL_FIGURES_DIR / "03_final_precision_recall_curve.png"

plt.savefig(
    pr_curve_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# SAVE FINAL TEST METRICS
# =========================================================

final_metrics = {
    "selected_model": selected_model_name,
    "cv_roc_auc": float(selected_cv_auc),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_roc_auc": float(test_roc_auc),
    "test_average_precision": float(test_average_precision),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}


final_metrics_path = MODEL_METRICS_DIR / "final_test_metrics.json"

with open(
    final_metrics_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_metrics,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ---------------------------------------------------------

classification_report_path = MODEL_METRICS_DIR / "classification_report.txt"

with open(
    classification_report_path,
    "w",
    encoding="utf-8",
) as file:
    file.write(report_text)


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

prediction_results = X_test[["PassengerId"]].copy()

prediction_results["Actual_Survived"] = y_test.values

prediction_results["Predicted_Survived"] = y_pred

prediction_results["Survival_Probability"] = y_prob


predictions_path = MODEL_METRICS_DIR / "test_predictions.csv"

prediction_results.to_csv(
    predictions_path,
    index=False,
)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model_path = MODELS_DIR / "titanic_random_forest_pipeline.joblib"

joblib.dump(
    selected_estimator,
    model_path,
)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("19. FINAL MODEL SUMMARY")
print("=" * 70)

print(f"\nSelected algorithm: {selected_model_name}")

print(f"Training CV ROC-AUC: {selected_cv_auc:.4f}")

print(f"Final test ROC-AUC:  {test_roc_auc:.4f}")

print(f"Final test accuracy: {test_accuracy:.4f}")

print(f"Final test F1 score: {test_f1:.4f}")


print("\nSaved figures:")

print(confusion_matrix_path)
print(roc_curve_path)
print(pr_curve_path)


print("\nSaved metrics:")

print(final_metrics_path)
print(classification_report_path)
print(predictions_path)


print("\nSaved model:")

print(model_path)


print("\n" + "=" * 70)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 70)

# =========================================================
# FEATURE IMPORTANCE ANALYSIS
# =========================================================

print("\n")
print("=" * 70)
print("20. FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# ACCESS FITTED PIPELINE COMPONENTS
# ---------------------------------------------------------

fitted_feature_engineer = selected_estimator.named_steps["week1_feature_engineering"]

fitted_preprocessor = selected_estimator.named_steps["column_preprocessing"]

fitted_model = selected_estimator.named_steps["model"]


# ---------------------------------------------------------
# TRANSFORMED FEATURE NAMES
# ---------------------------------------------------------

transformed_feature_names = fitted_preprocessor.get_feature_names_out()

feature_importances = fitted_model.feature_importances_


detailed_importance_df = pd.DataFrame(
    {
        "Feature": transformed_feature_names,
        "Importance": feature_importances,
    }
)


# Clean sklearn prefixes

detailed_importance_df["Feature"] = (
    detailed_importance_df["Feature"]
    .str.replace(
        "numeric__",
        "",
        regex=False,
    )
    .str.replace(
        "categorical__",
        "",
        regex=False,
    )
)


detailed_importance_df = detailed_importance_df.sort_values(
    "Importance",
    ascending=False,
).reset_index(drop=True)


print("\nDetailed transformed feature importance:\n")

print(
    detailed_importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------
# AGGREGATE ONE-HOT ENCODED CATEGORIES
# ---------------------------------------------------------


def get_original_feature(feature_name):

    if feature_name.startswith("Sex_"):
        return "Sex"

    if feature_name.startswith("Embarked_"):
        return "Embarked"

    return feature_name


detailed_importance_df["Original_Feature"] = detailed_importance_df["Feature"].apply(
    get_original_feature
)


aggregated_importance_df = (
    detailed_importance_df.groupby(
        "Original_Feature",
        as_index=False,
    )["Importance"]
    .sum()
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\nAggregated feature importance:\n")

print(
    aggregated_importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------
# SAVE FEATURE IMPORTANCE TABLES
# ---------------------------------------------------------

detailed_importance_path = MODEL_METRICS_DIR / "detailed_feature_importance.csv"

aggregated_importance_path = MODEL_METRICS_DIR / "aggregated_feature_importance.csv"


detailed_importance_df.to_csv(
    detailed_importance_path,
    index=False,
)

aggregated_importance_df.to_csv(
    aggregated_importance_path,
    index=False,
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE FIGURE
# ---------------------------------------------------------

plot_importance_df = aggregated_importance_df.sort_values(
    "Importance",
    ascending=True,
)


plt.figure(figsize=(9, 6))

plt.barh(
    plot_importance_df["Original_Feature"],
    plot_importance_df["Importance"],
)

plt.xlabel("Random Forest Importance")

plt.ylabel("Feature")

plt.title("Feature Importance - Final Random Forest Model")

plt.tight_layout()


feature_importance_path = MODEL_FIGURES_DIR / "04_feature_importance.png"

plt.savefig(
    feature_importance_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("\nSaved feature importance figure:")
print(feature_importance_path)


# =========================================================
# TEST-SET ERROR ANALYSIS
# =========================================================

print("\n")
print("=" * 70)
print("21. TEST-SET ERROR ANALYSIS")
print("=" * 70)


# Apply the fitted Week 1 feature-engineering stage
# to the holdout passengers.

engineered_test = fitted_feature_engineer.transform(X_test).copy()


analysis_df = engineered_test.copy()


analysis_df["PassengerId"] = X_test["PassengerId"].values


analysis_df["Actual_Survived"] = y_test.values


analysis_df["Predicted_Survived"] = y_pred


analysis_df["Survival_Probability"] = y_prob


# ---------------------------------------------------------
# DEFINE ERROR TYPE
# ---------------------------------------------------------

conditions = [
    ((analysis_df["Actual_Survived"] == 0) & (analysis_df["Predicted_Survived"] == 0)),
    ((analysis_df["Actual_Survived"] == 0) & (analysis_df["Predicted_Survived"] == 1)),
    ((analysis_df["Actual_Survived"] == 1) & (analysis_df["Predicted_Survived"] == 0)),
    ((analysis_df["Actual_Survived"] == 1) & (analysis_df["Predicted_Survived"] == 1)),
]


choices = [
    "True Negative",
    "False Positive",
    "False Negative",
    "True Positive",
]


analysis_df["Prediction_Type"] = np.select(
    conditions,
    choices,
    default="Unknown",
)


print("\nPrediction type distribution:\n")

print(analysis_df["Prediction_Type"].value_counts())


# ---------------------------------------------------------
# AGE GROUPS
# ---------------------------------------------------------

analysis_df["AgeGroup"] = pd.cut(
    analysis_df["Age"],
    bins=[
        0,
        12,
        18,
        35,
        60,
        np.inf,
    ],
    labels=[
        "Child",
        "Teen",
        "Young Adult",
        "Adult",
        "Senior",
    ],
    include_lowest=True,
)


# =========================================================
# SUBGROUP PERFORMANCE FUNCTION
# =========================================================


def calculate_subgroup_metrics(
    dataframe,
    group_column,
):

    rows = []

    grouped = dataframe.groupby(
        group_column,
        observed=False,
    )

    for group_name, group_df in grouped:
        if len(group_df) == 0:
            continue

        actual = group_df["Actual_Survived"]

        predicted = group_df["Predicted_Survived"]

        probabilities = group_df["Survival_Probability"]

        accuracy = accuracy_score(
            actual,
            predicted,
        )

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        if actual.nunique() == 2:
            auc = roc_auc_score(
                actual,
                probabilities,
            )

        else:
            auc = np.nan

        error_rate = (predicted != actual).mean()

        rows.append(
            {
                "Group_Type": group_column,
                "Group": str(group_name),
                "Passengers": len(group_df),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": auc,
                "Error_Rate": error_rate,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# SUBGROUPS TO ANALYZE
# =========================================================

subgroup_columns = [
    "Sex",
    "Pclass",
    "IsAlone",
    "Cabin_known",
    "AgeGroup",
]


subgroup_results = []


for column in subgroup_columns:
    result = calculate_subgroup_metrics(
        analysis_df,
        column,
    )

    subgroup_results.append(result)


subgroup_df = pd.concat(
    subgroup_results,
    ignore_index=True,
)


print("\nSubgroup performance:\n")

print(
    subgroup_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------
# SAVE SUBGROUP ANALYSIS
# ---------------------------------------------------------

subgroup_path = MODEL_METRICS_DIR / "subgroup_performance.csv"

subgroup_df.to_csv(
    subgroup_path,
    index=False,
)


analysis_path = MODEL_METRICS_DIR / "test_error_analysis.csv"

analysis_df.to_csv(
    analysis_path,
    index=False,
)


# =========================================================
# SUBGROUP ERROR-RATE FIGURE
# =========================================================

plot_subgroups = subgroup_df[subgroup_df["Passengers"] >= 5].copy()


plot_subgroups["Label"] = plot_subgroups["Group_Type"] + ": " + plot_subgroups["Group"]


plot_subgroups = plot_subgroups.sort_values(
    "Error_Rate",
    ascending=True,
)


plt.figure(figsize=(10, 8))

plt.barh(
    plot_subgroups["Label"],
    plot_subgroups["Error_Rate"],
)

plt.xlabel("Error Rate")

plt.ylabel("Passenger Subgroup")

plt.title("Prediction Error Rate Across Passenger Subgroups")

plt.tight_layout()


subgroup_figure_path = MODEL_FIGURES_DIR / "05_subgroup_error_rates.png"

plt.savefig(
    subgroup_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# MOST CONFIDENT ERRORS
# =========================================================

errors_df = analysis_df[
    analysis_df["Actual_Survived"] != analysis_df["Predicted_Survived"]
].copy()


errors_df["Prediction_Confidence"] = np.where(
    errors_df["Predicted_Survived"] == 1,
    errors_df["Survival_Probability"],
    1 - errors_df["Survival_Probability"],
)


errors_df = errors_df.sort_values(
    "Prediction_Confidence",
    ascending=False,
)


confident_errors_path = MODEL_METRICS_DIR / "most_confident_errors.csv"

errors_df.to_csv(
    confident_errors_path,
    index=False,
)


print("\nTotal test errors:")
print(len(errors_df))


print("\nMost confident incorrect predictions:\n")

print(
    errors_df[
        [
            "PassengerId",
            "Sex",
            "Pclass",
            "Age",
            "FamilySize",
            "IsAlone",
            "Cabin_known",
            "Actual_Survived",
            "Predicted_Survived",
            "Survival_Probability",
            "Prediction_Type",
        ]
    ]
    .head(10)
    .to_string(
        index=False,
    )
)


print("\nSaved subgroup analysis:")
print(subgroup_path)

print("\nSaved complete test error analysis:")
print(analysis_path)

print("\nSaved confident errors:")
print(confident_errors_path)

print("\nSaved subgroup figure:")
print(subgroup_figure_path)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE AND ERROR ANALYSIS COMPLETE")
print("=" * 70)

# =========================================================
# FEATURE ENGINEERING EFFECTIVENESS
# =========================================================

print("\n")
print("=" * 70)
print("22. FEATURE ENGINEERING EFFECTIVENESS")
print("=" * 70)


class TitanicBasicFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Creates a basic Titanic feature set without the additional
    Week 1 engineered variables.

    This is used only for comparison with the full Week 1
    feature-engineering approach.
    """

    def fit(self, X, y=None):

        X = X.copy()

        self.age_group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median()

        self.age_global_median_ = X["Age"].median()

        self.embarked_mode_ = X["Embarked"].mode()[0]

        self.fare_median_ = X["Fare"].median()

        return self

    def transform(self, X):

        X = X.copy()

        # ---------------------------------
        # Age imputation
        # ---------------------------------

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

        # ---------------------------------
        # Embarked
        # ---------------------------------

        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)

        # ---------------------------------
        # Fare
        # ---------------------------------

        X["Fare"] = X["Fare"].fillna(self.fare_median_)

        basic_features = [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex",
            "Embarked",
        ]

        return X[basic_features]


# ---------------------------------------------------------
# BASIC FEATURE PREPROCESSOR
# ---------------------------------------------------------

basic_numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

basic_categorical_features = [
    "Sex",
    "Embarked",
]


basic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            basic_numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            basic_categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# SAME RANDOM FOREST FOR BOTH FEATURE SETS
# ---------------------------------------------------------


def create_comparison_forest():

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


# ---------------------------------------------------------
# BASIC PIPELINE
# ---------------------------------------------------------

basic_pipeline = Pipeline(
    steps=[
        (
            "basic_feature_engineering",
            TitanicBasicFeatureEngineer(),
        ),
        (
            "preprocessing",
            basic_preprocessor,
        ),
        (
            "model",
            create_comparison_forest(),
        ),
    ]
)


# ---------------------------------------------------------
# ENGINEERED PIPELINE
# ---------------------------------------------------------

engineered_pipeline = Pipeline(
    steps=[
        (
            "week1_feature_engineering",
            TitanicFeatureEngineer(),
        ),
        (
            "preprocessing",
            column_preprocessor,
        ),
        (
            "model",
            create_comparison_forest(),
        ),
    ]
)


# ---------------------------------------------------------
# SAME CV SPLITS AND SAME METRICS
# ---------------------------------------------------------

feature_comparison_scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


basic_scores = cross_validate(
    basic_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=feature_comparison_scoring,
    n_jobs=1,
)


engineered_scores = cross_validate(
    engineered_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=feature_comparison_scoring,
    n_jobs=1,
)


feature_comparison_df = pd.DataFrame(
    [
        {
            "Feature_Set": "Basic Features",
            "CV_Accuracy": (basic_scores["test_accuracy"].mean()),
            "CV_F1": (basic_scores["test_f1"].mean()),
            "CV_ROC_AUC": (basic_scores["test_roc_auc"].mean()),
            "ROC_AUC_Std": (basic_scores["test_roc_auc"].std()),
        },
        {
            "Feature_Set": "Week 1 Engineered Features",
            "CV_Accuracy": (engineered_scores["test_accuracy"].mean()),
            "CV_F1": (engineered_scores["test_f1"].mean()),
            "CV_ROC_AUC": (engineered_scores["test_roc_auc"].mean()),
            "ROC_AUC_Std": (engineered_scores["test_roc_auc"].std()),
        },
    ]
)


print("\nControlled feature-set comparison:\n")

print(
    feature_comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


basic_auc = feature_comparison_df.loc[
    feature_comparison_df["Feature_Set"] == "Basic Features",
    "CV_ROC_AUC",
].iloc[0]


engineered_auc = feature_comparison_df.loc[
    feature_comparison_df["Feature_Set"] == "Week 1 Engineered Features",
    "CV_ROC_AUC",
].iloc[0]


feature_auc_improvement = engineered_auc - basic_auc


print("\nROC-AUC improvement from Week 1 feature engineering:")

print(f"{feature_auc_improvement:+.4f}")


feature_comparison_path = MODEL_METRICS_DIR / "feature_engineering_comparison.csv"


feature_comparison_df.to_csv(
    feature_comparison_path,
    index=False,
)


# ---------------------------------------------------------
# FEATURE ENGINEERING COMPARISON FIGURE
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    feature_comparison_df["Feature_Set"],
    feature_comparison_df["CV_ROC_AUC"],
)

plt.ylabel("Mean 5-Fold CV ROC-AUC")

plt.xlabel("Feature Set")

plt.title("Effect of Week 1 Feature Engineering")

plt.ylim(
    0.70,
    1.00,
)

plt.tight_layout()


feature_comparison_figure_path = (
    MODEL_FIGURES_DIR / "06_feature_engineering_comparison.png"
)


plt.savefig(
    feature_comparison_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# BASELINE VS TUNED MODEL COMPARISON
# =========================================================

print("\n")
print("=" * 70)
print("23. BASELINE VS TUNED MODEL PERFORMANCE")
print("=" * 70)


baseline_auc_df = comparison_df[
    [
        "Model",
        "ROC-AUC Mean",
    ]
].copy()


baseline_auc_df = baseline_auc_df.rename(columns={"ROC-AUC Mean": "Baseline_ROC_AUC"})


tuned_auc_df = tuned_comparison_df[
    [
        "Model",
        "CV ROC-AUC",
    ]
].copy()


tuned_auc_df = tuned_auc_df.rename(columns={"CV ROC-AUC": "Tuned_ROC_AUC"})


baseline_tuned_df = baseline_auc_df.merge(
    tuned_auc_df,
    on="Model",
)


baseline_tuned_df["Improvement"] = (
    baseline_tuned_df["Tuned_ROC_AUC"] - baseline_tuned_df["Baseline_ROC_AUC"]
)


print("\nBaseline versus tuned ROC-AUC:\n")

print(
    baseline_tuned_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


baseline_tuned_path = MODEL_METRICS_DIR / "baseline_vs_tuned.csv"


baseline_tuned_df.to_csv(
    baseline_tuned_path,
    index=False,
)


# ---------------------------------------------------------
# MODEL TUNING FIGURE
# ---------------------------------------------------------

plot_df = baseline_tuned_df.set_index("Model")[
    [
        "Baseline_ROC_AUC",
        "Tuned_ROC_AUC",
    ]
]


ax = plot_df.plot(
    kind="bar",
    figsize=(9, 6),
)

ax.set_ylabel("Mean CV ROC-AUC")

ax.set_xlabel("Model")

ax.set_title("Baseline vs Tuned Model Performance")

ax.set_ylim(
    0.70,
    1.00,
)

plt.xticks(
    rotation=0,
)

plt.tight_layout()


tuning_figure_path = MODEL_FIGURES_DIR / "07_baseline_vs_tuned_models.png"


plt.savefig(
    tuning_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# PROBABILITY CALIBRATION
# =========================================================

print("\n")
print("=" * 70)
print("24. PROBABILITY CALIBRATION")
print("=" * 70)


prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=8,
    strategy="quantile",
)


brier_score = brier_score_loss(
    y_test,
    y_prob,
)


print(f"\nBrier Score: {brier_score:.4f}")

print("Lower Brier scores indicate better probability accuracy.")


calibration_df = pd.DataFrame(
    {
        "Mean_Predicted_Probability": prob_pred,
        "Observed_Survival_Rate": prob_true,
    }
)


print("\nCalibration data:\n")

print(
    calibration_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


calibration_path = MODEL_METRICS_DIR / "calibration_data.csv"


calibration_df.to_csv(
    calibration_path,
    index=False,
)


# ---------------------------------------------------------
# CALIBRATION FIGURE
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Random Forest",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration",
)

plt.xlabel("Mean Predicted Survival Probability")

plt.ylabel("Observed Survival Frequency")

plt.title("Calibration Curve - Final Random Forest")

plt.legend()

plt.tight_layout()


calibration_figure_path = MODEL_FIGURES_DIR / "08_calibration_curve.png"


plt.savefig(
    calibration_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# FINAL ANALYSIS SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("25. COMPLETE WEEK 4 ANALYSIS SUMMARY")
print("=" * 70)


print(f"\nSelected model: {selected_model_name}")

print(f"Tuned CV ROC-AUC: {selected_cv_auc:.4f}")

print(f"Holdout ROC-AUC: {test_roc_auc:.4f}")

print(f"Holdout Accuracy: {test_accuracy:.4f}")

print(f"Holdout F1: {test_f1:.4f}")

print(f"Holdout Recall: {test_recall:.4f}")

print(f"Feature engineering ROC-AUC change: {feature_auc_improvement:+.4f}")

print(f"Brier Score: {brier_score:.4f}")


print("\nAdditional figures saved:")

print(feature_comparison_figure_path)

print(tuning_figure_path)

print(calibration_figure_path)


print(
    "\nThe final test set was used only for "
    "final evaluation and post-hoc diagnostic "
    "analysis. No further hyperparameter tuning "
    "was performed using test-set results."
)


print("\n" + "=" * 70)
print("WEEK 4 MODEL ANALYSIS COMPLETE")
print("=" * 70)
