
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SEED

TEST_SIZE = 0.2
N_ESTIMATORS = 200
N_FOLDS = 5
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def cross_validate_model(X, y):
    """Stratified k-fold CV across the whole dataset, to check whether a high
    single-split score is real signal or overfitting on a small, high-dimensional
    feature set."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    scores = cross_validate(clf, X, y, cv=cv, scoring=CV_SCORING)

    print(f"{N_FOLDS}-fold cross-validation (stratified):")
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

    cross_validate_model(X, y)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    clf.fit(X_train, y_train)

    importances = clf.feature_importances_
    top10 = np.argsort(importances)[-10:][::-1]
    print("Top 10 most important features:")
    for i in top10:
        print(f"  {X.columns[i]}: {importances[i]:.4f}")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    if y_test.nunique() > 1:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    else:
        print("ROC-AUC:   undefined (test set has only one class)")

    print("\nConfusion matrix (rows=actual, cols=predicted; order=[real, fake]):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["real", "fake"], zero_division=0))

    return clf


if __name__ == "__main__":
    train_and_evaluate()
    