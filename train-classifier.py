
import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SEED

TEST_SIZE = 0.2
N_ESTIMATORS = 200
N_FOLDS = 5
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def build_models():
    """Fresh, unfitted model instances. SVM and Logistic Regression get a
    StandardScaler first since they're distance/gradient-based and sensitive to
    feature scale, unlike the tree-based models (Random Forest, XGBoost), which
    split on raw thresholds and don't need scaling."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, eval_metric="logloss"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "SVM (linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", random_state=SEED)),
        ]),
    }


def get_decision_scores(model, X_test):
    """Per-sample scores for ROC-AUC -- predict_proba where available, falling
    back to decision_function (e.g. for an SVM without probability=True)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def get_top_features(model, feature_names, n=10):
    """Top-n most influential features, handling both tree-based models
    (feature_importances_) and linear models (coef_, ranked by magnitude since
    sign just indicates direction toward real vs. fake)."""
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    else:
        return []

    top_idx = np.argsort(scores)[-n:][::-1]
    return [(feature_names[i], scores[i]) for i in top_idx]


def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: real", "Actual: fake"],
        columns=["Predicted: real", "Predicted: fake"],
    )
    print(cm_df.to_string())

def cross_validate_models(X, y):
    """Stratified k-fold CV across the whole dataset for every model, to check
    whether a high single-split score is real signal or overfitting on a small,
    high-dimensional feature set."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for name, model in build_models().items():
        print(f"\n=== {name} -- {N_FOLDS}-fold cross-validation (stratified) ===")
        scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING)
        for metric in CV_SCORING:
            fold_scores = scores[f"test_{metric}"]
            per_fold = ", ".join(f"{s:.3f}" for s in fold_scores)
            print(f"  {metric:10s}: mean={fold_scores.mean():.3f} std={fold_scores.std():.3f}  (folds: {per_fold})")

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols]
    y = (df["label"] == "fake").astype(int)  # 1 = fake, 0 = real
    return X, y


def train_and_evaluate():
    X, y = load_features()
    print(f"Total number of features: {X.shape[1]}")

    cross_validate_models(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    fitted_models = {}
    for name, model in build_models().items():
        print(f"\n===== {name} =====")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        top_features = get_top_features(model, X.columns)
        if top_features:
            print("Top 10 most important features:")
            for feat_name, score in top_features:
                print(f"  {feat_name}: {score:.4f}")
        else:
            print("Top 10 most important features: not available for this model")

        y_pred = model.predict(X_test)
        y_scores = get_decision_scores(model, X_test)

        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
        print(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
        if y_test.nunique() > 1:
            print(f"ROC-AUC:   {roc_auc_score(y_test, y_scores):.3f}")
        else:
            print("ROC-AUC:   undefined (test set has only one class)")

        print("\nConfusion matrix:")
        print_confusion_matrix(y_test, y_pred)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["real", "fake"], zero_division=0))

    return fitted_models


if __name__ == "__main__":
    train_and_evaluate()
    