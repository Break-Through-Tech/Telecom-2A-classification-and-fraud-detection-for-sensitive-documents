
import os
from pathlib import Path
from sklearn.base import clone
import sys

import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import PredefinedSplit, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_PATH, SEED, N_FOLDS
N_ESTIMATORS = 200
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

def build_models():
    """Fresh, unfitted model instances. SVM and Logistic Regression get a
    StandardScaler first since they're distance/gradient-based and sensitive to
    feature scale, unlike the tree-based models (Random Forest, XGBoost), which
    split on raw thresholds and don't need scaling."""
    return {
        "Extra Trees": ExtraTreesClassifier(n_estimators=600, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, verbosity=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "SVM (linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", random_state=SEED)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=SEED)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=SEED)),
        ]),
    }


def get_decision_scores(model, X_test):
    """Per-sample scores for ROC-AUC -- predict_proba where available, falling
    back to decision_function (e.g. for an SVM without probability=True)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)

def with_feature_selection(estimator, count):
    """Fit a fresh supervised selector whenever a CV pipeline is fitted."""
    return Pipeline([
        ("select", SelectFromModel(
            ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
            threshold=-np.inf, max_features=count,
        )),
        ("clf", estimator),
    ])


def selected_feature_names(model, feature_names):
    names = np.asarray(feature_names, dtype=object)
    estimator = model
    while hasattr(estimator, "steps"):
        for _, step in estimator.steps[:-1]:
            if hasattr(step, "get_support"):
                names = names[step.get_support()]
        estimator = estimator.steps[-1][1]
    return names, estimator


def write_selected_features(model, feature_names, features_path):
    selector = model.named_steps["select"]
    names, final_estimator = selected_feature_names(model, feature_names)
    chosen = selector.get_support()
    scores = np.asarray(selector.estimator_.feature_importances_)[chosen]
    report = pd.DataFrame({"feature": names, "selector_importance": scores})
    if hasattr(final_estimator, "feature_importances_"):
        report["classifier_importance"] = final_estimator.feature_importances_
    report["family"] = report["feature"].map(
        lambda name: name.split("_")[1] if name.startswith("feat_") else "other"
    )
    report = report.sort_values("selector_importance", ascending=False).reset_index(drop=True)
    report.insert(0, "rank", np.arange(1, len(report) + 1))
    source = Path(features_path or FEATURES_PATH).resolve()
    destination = source.with_name(source.stem + f"_top{len(report)}_selected_features.csv")
    temporary = destination.with_suffix(".csv.tmp")
    report.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"\nFinal all-development fit uses {len(report)} selected features:")
    print(report.groupby("family").size().rename("count").to_string())
    print(f"Selected feature list: {destination}")
    print("Selection is refitted within every CV training fold; this list belongs only to the final all-development fit.")
    return destination


def get_top_features(model, feature_names, n=10):
    """Map importance values to the selected columns, including nested pipelines."""
    names, estimator = selected_feature_names(model, feature_names)
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    else:
        return []

    top_idx = np.argsort(scores)[-n:][::-1]
    return [(names[i], scores[i]) for i in top_idx]


def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: real", "Actual: fake"],
        columns=["Predicted: real", "Predicted: fake"],
    )
    print(cm_df.to_string())

def cross_validate_models(X, y, fold_ids, models=None):
    """Compare models on development sources only, using saved source folds."""
    cv = PredefinedSplit(fold_ids.to_numpy())
    summary = {}
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    for name, model in (models if models is not None else build_models()).items():
        print(f"\n=== {name} -- {N_FOLDS}-fold development CV (source-separated) ===")
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")
        summary[name] = {}
        for metric in CV_SCORING:
            fold_scores = scores[f"test_{metric}"]
            summary[name][metric] = float(fold_scores.mean())
            per_fold = ", ".join(f"{s:.3f}" for s in fold_scores)
            print(f"  {metric:10s}: mean={fold_scores.mean():.3f} std={fold_scores.std():.3f}  (folds: {per_fold})")
    return summary

def load_features(feature_set="all", features_path=None):
    df = pd.read_csv(features_path or FEATURES_PATH)
    required = {"label", "source_id", "fold_id", "split", "donor_id", "tamper_type"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Feature CSV {Path(features_path or FEATURES_PATH).resolve()} is missing metadata columns: {missing}")
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if feature_set == "forensic":
        feature_cols = [c for c in feature_cols if not c.startswith("feat_hog_")]
    elif feature_set == "hog":
        feature_cols = [c for c in feature_cols if c.startswith("feat_hog_")]
    elif feature_set != "all":
        raise ValueError(f"Unknown feature set: {feature_set}")
    if not feature_cols or df.empty:
        raise ValueError("No feature rows/columns found")
    X = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains missing or infinite values")
    # Keep the generator's string labels; accept old numeric 0/1 explicitly too.
    y = df["label"].astype(str).str.strip().str.lower().map({
        "real": 0, "fake": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1,
    })
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Expected both real/fake classes (or numeric 0/1)")
    y = y.astype(int)
    folds = pd.to_numeric(df["fold_id"], errors="raise")
    if not folds.isin([-1, *range(N_FOLDS)]).all():
        raise ValueError("Invalid fold_id; generator/classifier N_FOLDS must match")
    df["fold_id"] = folds.astype(int)
    if not df["split"].isin(["development", "test"]).all():
        raise ValueError("Invalid split labels")
    if not df["split"].eq("test").equals(df["fold_id"].eq(-1)):
        raise ValueError("Only test rows must have fold_id=-1")
    if df["source_id"].isna().any():
        raise ValueError("Missing source IDs")
    if df.groupby("source_id")["fold_id"].nunique().gt(1).any():
        raise ValueError("A source crosses an evaluation boundary")
    if not y.groupby(df["source_id"]).nunique().eq(2).all():
        raise ValueError("Each source must have both real and fake examples")
    source_folds = df.drop_duplicates("source_id").set_index("source_id")["fold_id"]
    for _, row in df.loc[df["tamper_type"] == "splice"].iterrows():
        donor = row["donor_id"]
        if donor == row["source_id"] or donor not in source_folds or source_folds[donor] != row["fold_id"]:
            raise ValueError("Splice donor is missing, identical, or crosses an evaluation boundary")
    return X, y, df

def tune_models(models, summary, X, y, folds, metric, iterations=12):
    """Small randomized searches on saved development folds only.

    These are model-selection scores; the reserved test sources remain separate.
    The baseline winner is retained unless a search improves its CV score.
    """
    spaces = {
        "Extra Trees": {
            "n_estimators": [300, 600, 900],
            "max_features": ["sqrt", 0.2, 0.5, 1.0],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 15, 30],
        },
        "Random Forest": {
            "n_estimators": [300, 600, 900],
            "max_features": ["sqrt", 0.2, 0.5],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 15, 30],
        },
        "SVM (RBF)": {
            "clf__C": [0.1, 1, 10, 100, 1000],
            "clf__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1],
        },
        "SVM (linear)": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "XGBoost": {
            "n_estimators": [200, 400, 600], "max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [1, 5, 10],
        },
        "LightGBM": {
            "n_estimators": [200, 400, 600], "num_leaves": [7, 15, 31],
            "learning_rate": [0.03, 0.1], "min_child_samples": [10, 20, 40],
            "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0, 1, 5],
        },
    }
    best_name = max(summary, key=lambda name: summary[name][metric])
    best_model = models[best_name]
    best_score = summary[best_name][metric]
    candidates = sorted(
        (name for name in models if name in spaces),
        key=lambda name: summary[name][metric], reverse=True,
    )[:2]
    if not candidates:
        raise ValueError("No tuning grid for this model. Choose Extra Trees, Random Forest, SVM, Logistic Regression, XGBoost, or LightGBM.")
    scoring = make_scorer(f1_score, zero_division=0) if metric == "f1" else "accuracy"
    for name in candidates:
        parameter_count = 1
        for values in spaces[name].values():
            parameter_count *= len(values)
        print(f"\nTuning {name} for development {metric}...")
        parameters = spaces[name]
        if hasattr(models[name], "named_steps") and "select" in models[name].named_steps:
            parameters = {"clf__" + key: values for key, values in parameters.items()}
        search = RandomizedSearchCV(
            models[name], parameters, n_iter=min(iterations, parameter_count),
            scoring=scoring, cv=PredefinedSplit(folds.to_numpy()),
            random_state=SEED, n_jobs=1, refit=True, error_score="raise",
        )
        search.fit(X, y)
        print(f"  Best mean CV {metric}: {search.best_score_:.4f}")
        print(f"  Parameters: {search.best_params_}")
        if search.best_score_ > best_score:
            best_name, best_model, best_score = name, search.best_estimator_, float(search.best_score_)
    print(f"Selected development {metric}: {best_score:.4f} (used for model selection)")
    return best_name, best_model

def choose_probability_threshold(y, scores, metric="accuracy"):
    """Choose a cutoff from predictions made without fitting on those rows."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if metric not in ("accuracy", "f1"):
        raise ValueError("Threshold metric must be accuracy or f1")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Threshold tuning requires finite fake-class scores in [0, 1]")
    scorer = accuracy_score if metric == "accuracy" else lambda a,b: f1_score(a,b,zero_division=0)
    candidates = np.round(np.arange(0.10, 0.901, 0.01), 2)
    # Equal scores favor the cutoff nearest 0.5, then the more conservative one.
    best = max(candidates, key=lambda t:(scorer(y, scores > t), -abs(t-.5), t))
    return float(best), float(scorer(y, scores > best))


def fake_probability(model, X):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Choose a probabilistic model such as Extra Trees for threshold tuning")
    positions = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positions) != 1:
        raise ValueError("Expected binary real/fake class labels 0 and 1")
    return np.asarray(model.predict_proba(X))[:, positions[0]]


def prediction_metrics(y, predictions, scores):
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
    }


def nested_threshold_evaluation(estimator, X, y, folds, metric):
    """Outer saved folds assess cutoffs selected using inner saved folds.

    Model identity and hyperparameters must be fixed for this procedure.
    Existing fold IDs keep source images and splice donors together at both levels.
    """
    count = len(y)
    probabilities = np.full(count, np.nan)
    baseline = np.zeros(count, dtype=int)
    adjusted = np.zeros(count, dtype=int)
    thresholds = np.full(count, np.nan)
    seen = np.zeros(count, dtype=bool)
    results = []
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("--tune-threshold needs predict_proba; use Extra Trees or another probability model")
    outer_cv = PredefinedSplit(folds.to_numpy())
    for train, valid in outer_cv.split():
        if seen[valid].any():
            raise ValueError("Overlapping outer validation folds")
        inner_folds = folds.iloc[train]
        inner_cv = PredefinedSplit(inner_folds.to_numpy())
        if inner_folds.nunique() < 2:
            raise ValueError("Need at least two source-separated inner folds")
        inner_scores = cross_val_predict(
            clone(estimator), X.iloc[train], y.iloc[train],
            cv=inner_cv, method="predict_proba", n_jobs=1,
        )[:, 1]  # Binary labels are validated as 0/1 by load_features.
        cutoff, inner_score = choose_probability_threshold(y.iloc[train], inner_scores, metric)
        fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
        scores = fake_probability(fitted, X.iloc[valid])
        # Match sklearn's default strictly-greater-than 0.5 binary decision.
        normal = (scores > 0.5).astype(int)
        tuned = (scores > cutoff).astype(int)
        probabilities[valid], baseline[valid], adjusted[valid], thresholds[valid] = scores, normal, tuned, cutoff
        seen[valid] = True
        record = {"fold_id": int(folds.iloc[valid[0]]), "threshold": cutoff,
                  "inner_selection_score": inner_score}
        for prefix, predictions in (("default", normal), ("adjusted", tuned)):
            record.update({f"{prefix}_{name}": value for name,value in
                           prediction_metrics(y.iloc[valid], predictions, scores).items()})
        results.append(record)
        print(f"Outer fold {record['fold_id']}: cutoff={cutoff:.2f}, accuracy {record['default_accuracy']:.3f} -> {record['adjusted_accuracy']:.3f}")
    if not seen.all() or not np.isfinite(probabilities).all():
        raise ValueError("Every development example must receive an outer-fold prediction")
    summary = pd.DataFrame(results)
    print("\nSource-separated nested threshold evaluation (fixed model/settings):")
    for name in CV_SCORING:
        before, after = summary[f"default_{name}"], summary[f"adjusted_{name}"]
        print(f"  {name:10s}: default={before.mean():.3f} +/- {before.std(ddof=0):.3f}; adjusted={after.mean():.3f} +/- {after.std(ddof=0):.3f}")
    print("ROC-AUC is unchanged by moving the cutoff; false-positive tradeoffs matter.")
    # Used only for the final all-development fit / optional held-out test.
    # Do not report this cutoff's score on these same OOF rows as independent CV.
    final_threshold, _ = choose_probability_threshold(y, probabilities, metric)
    print(f"Final cutoff for the all-development model: {final_threshold:.2f} (chosen using development OOF scores only)")
    return probabilities, baseline, adjusted, thresholds, summary, final_threshold


def write_threshold_audit(metadata, y, probabilities, baseline, adjusted, thresholds, features_path, max_features=None):
    audit = metadata.loc[:, [c for c in metadata if not c.startswith("feat_")]].reset_index(drop=True).copy()
    audit["actual_fake"] = np.asarray(y, dtype=int)
    audit["fake_score"] = probabilities
    audit["outer_threshold"] = thresholds
    audit["default_prediction"] = baseline
    audit["adjusted_prediction"] = adjusted
    audit["default_correct"] = baseline == np.asarray(y)
    audit["adjusted_correct"] = adjusted == np.asarray(y)
    audit["adjusted_error"] = np.where(
        (np.asarray(y) == 1) & (adjusted == 0), "missed_fake",
        np.where((np.asarray(y) == 0) & (adjusted == 1), "flagged_real", "correct"),
    )
    source = Path(features_path or FEATURES_PATH).resolve()
    suffix = f"_top{max_features}" if max_features is not None else ""
    destination = source.with_name(source.stem + suffix + "_threshold_audit.csv")
    temporary = destination.with_suffix(".csv.tmp")
    audit.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(f"Development error audit: {destination} (diagnostic CSV is replaced on rerun)")
    print("\nOuter-fold fake recall by tamper type:")
    fake = audit.loc[audit["actual_fake"] == 1]
    print(fake.groupby("tamper_type").agg(
        count=("actual_fake", "size"), default_recall=("default_prediction", "mean"),
        adjusted_recall=("adjusted_prediction", "mean"),
    ).to_string())
    print("\nOuter-fold error counts by saved fold:")
    print(pd.crosstab(audit["fold_id"], audit["adjusted_error"]).to_string())
    print("\nReal examples incorrectly flagged:")
    real = audit.loc[audit["actual_fake"] == 0]
    print(f"Default: {int(real['default_prediction'].sum())}; adjusted: {int(real['adjusted_prediction'].sum())}")
    return destination


def train_and_evaluate(feature_set="all", evaluate_test=False, tune=False,
                       metric="f1", model_name=None, search_iterations=12,
                       features_path=None, tune_threshold=False, max_features=None):
    if tune_threshold and tune:
        raise ValueError("Run threshold selection separately from --tune; nested threshold evaluation requires fixed hyperparameters")
    if tune_threshold and model_name is None:
        model_name = "Extra Trees"
    if metric not in ("accuracy", "f1"):
        raise ValueError("metric must be accuracy or f1")
    if search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    X, y, metadata = load_features(feature_set, features_path)
    if max_features is not None and not 1 <= max_features <= X.shape[1]:
        raise ValueError(f"--max-features must be between 1 and {X.shape[1]}")
    development = metadata["split"].eq("development")
    held_out = metadata["split"].eq("test")
    X_train, y_train = X.loc[development], y.loc[development]
    X_test, y_test = X.loc[held_out], y.loc[held_out]
    folds = metadata.loc[development, "fold_id"]
    if set(folds.unique()) != set(range(N_FOLDS)) or X_test.empty:
        raise ValueError("Need every development fold and a separate test partition")
    for fold in range(N_FOLDS):
        if y_train.loc[folds.eq(fold)].nunique() != 2:
            raise ValueError(f"Development fold {fold} must contain both classes")
    if y_test.nunique() != 2:
        raise ValueError("Test partition must contain both classes")

    print(f"Total features: {X.shape[1]}; development: {len(X_train)}; test: {len(X_test)}")
    models = build_models()
    if model_name:
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        models = {model_name: models[model_name]}
    if max_features is not None:
        models = {name: with_feature_selection(estimator, max_features) for name, estimator in models.items()}
        print(f"Feature selection enabled: {X.shape[1]} input columns -> {max_features} per fitted model")
        print("Selectors are learned inside each training fold, including inner threshold folds.")
    final_threshold = None
    if tune_threshold:
        selected_model = models[model_name]
        best_name = model_name
        scores, normal_oof, oof, cutoff_per_row, fold_summary, final_threshold = nested_threshold_evaluation(
            selected_model, X_train, y_train, folds, metric,
        )
        write_threshold_audit(
            metadata.loc[development], y_train, scores, normal_oof, oof,
            cutoff_per_row, features_path, max_features,
        )
        print(f"Threshold experiment model: {best_name}; objective: {metric}")
    else:
        summary = cross_validate_models(X_train, y_train, folds, models)
        best_name = max(summary, key=lambda name: summary[name][metric])
        selected_model = models[best_name]
        if tune:
            best_name, selected_model = tune_models(
                models, summary, X_train, y_train, folds, metric, search_iterations,
            )
        print(f"\nSelected by development {metric}: {best_name}")
        print(f"Feature family: {feature_set}")

        # Out-of-fold development predictions help diagnose tamper types without
        # repeatedly inspecting the reserved test set. Selection used these same
        # folds, so this diagnostic is not an independent final performance estimate.
        oof = cross_val_predict(
            selected_model, X_train, y_train,
            cv=PredefinedSplit(folds.to_numpy()), method="predict",
         )

    audit = metadata.loc[development, ["tamper_type"]].copy()
    audit["actual_fake"] = y_train.to_numpy()
    audit["detected"] = oof == 1
    print("\nDevelopment OOF fake recall by tamper type (after model selection):")
    print(audit.loc[audit["actual_fake"] == 1].groupby("tamper_type")["detected"]
          .agg(count="size", recall="mean").to_string())
    model = selected_model
    model.fit(X_train, y_train)
    if max_features is not None:
        write_selected_features(model, X.columns, features_path)
    if final_threshold is not None:
        model.decision_threshold_ = final_threshold
    fitted_models = {best_name: model}
    if not evaluate_test:
        print("Test set untouched. Use --evaluate-test once settings are finalized.")
        return fitted_models

    model = fitted_models[best_name]
    top_features = get_top_features(model, X.columns)
    if top_features:
        print("Top 10 most influential features for the selected model:")
        for feat_name, score in top_features:
            print(f"  {feat_name}: {score:.4f}")

    y_scores = get_decision_scores(model, X_test)
    y_pred = (y_scores > final_threshold).astype(int) if final_threshold is not None else model.predict(X_test)
    if final_threshold is not None:
        print(f"Using the development-selected cutoff {final_threshold:.2f} for held-out testing")
    print(f"\n===== Held-out test: {best_name} =====")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_scores):.3f}")
    print("\nConfusion matrix:")
    print_confusion_matrix(y_test, y_pred)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["real", "fake"], zero_division=0))

    audit = metadata.loc[held_out, ["tamper_type"]].copy()
    audit["actual_fake"] = y_test.to_numpy()
    audit["detected"] = np.asarray(y_pred) == 1
    print("\nFake detection recall by tamper type (test):")
    print(audit.loc[audit["actual_fake"] == 1].groupby("tamper_type")["detected"]
          .agg(count="size", recall="mean").to_string())
    print("Repeated changes based on these test results make the test set part of development.")

    return fitted_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=("all", "forensic", "hog"), default="all")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the selected model on reserved test sources")
    parser.add_argument("--tune", action="store_true", help="Tune up to two development leaders; may take substantially longer")
    parser.add_argument("--metric", choices=("accuracy", "f1"), default="f1")
    parser.add_argument("--model", choices=list(build_models()), help="Compare/tune only one model to save runtime")
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--features-file", help="Use an existing clean feature CSV without rebuilding images")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Nested source-separated cutoff selection; defaults to Extra Trees and leaves test untouched")
    parser.add_argument("--max-features", type=int,
                        help="Fit a selector inside every CV fold, e.g. --max-features 150; default uses all input features")
    args = parser.parse_args()
    train_and_evaluate(
        feature_set=args.feature_set,
        evaluate_test=args.evaluate_test,
        tune=args.tune,
        metric=args.metric,
        model_name=args.model,
        search_iterations=args.search_iterations,
        features_path=args.features_file,
        tune_threshold=args.tune_threshold,
        max_features=args.max_features,
    )