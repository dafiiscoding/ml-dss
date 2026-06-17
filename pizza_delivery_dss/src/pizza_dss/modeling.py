"""Huấn luyện & đánh giá mô hình dự báo trễ.

Quy trình cho người mới: fit trên train -> chọn mô hình trên dev theo **F2** ->
khoá -> báo test 1 lần (kèm bootstrap CI). `cross_validate_models` kiểm độ ổn
định; `build_model_figures` vẽ confusion matrix/ROC/so sánh.

Ghi chú: nếu máy chặn DLL native của sklearn (WDAC), các lớp `Simple*Classifier`
thuần Python được dùng thay thế tự động qua `_optional_classifier`. Thuật ngữ
(F2, MCC, ROC-AUC, CV, bootstrap…) xem `docs/GLOSSARY.md`.
"""
import json
import importlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pizza_dss.config import (
    ACTIVE_CATEGORICAL_FEATURES,
    ACTIVE_FEATURE_COLUMNS,
    ACTIVE_NUMERIC_FEATURES,
    COMPACT_CATEGORICAL_FEATURES,
    COMPACT_FEATURE_COLUMNS,
    COMPACT_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from pizza_dss.data_loader import load_processed_splits, validate_feature_contract
from pizza_dss.features import build_preprocessor


def _dense_array(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X, dtype=float)


class SimpleDecisionStumpClassifier(BaseEstimator, ClassifierMixin):
    """Small pure-Python fallback when WDAC blocks sklearn native tree DLLs."""

    def __init__(self, class_weight=None, random_state=None):
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X = _dense_array(X)
        y = np.asarray(y, dtype=bool)
        self.classes_ = np.array([False, True])
        self.base_probability_ = float(y.mean()) if len(y) else 0.0
        self.feature_index_ = 0
        self.threshold_ = 0.0
        self.left_probability_ = self.base_probability_
        self.right_probability_ = self.base_probability_

        if X.size == 0 or y.min() == y.max():
            return self

        sample_weight = np.ones(len(y), dtype=float)
        if self.class_weight == "balanced":
            positive = max(int(y.sum()), 1)
            negative = max(len(y) - positive, 1)
            sample_weight[y] = len(y) / (2 * positive)
            sample_weight[~y] = len(y) / (2 * negative)

        best_loss = float("inf")
        for feature_index in range(X.shape[1]):
            values = X[:, feature_index]
            thresholds = np.unique(np.nanpercentile(values, [20, 40, 60, 80]))
            for threshold in thresholds:
                left = values <= threshold
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                loss = self._weighted_gini(y[left], sample_weight[left]) * sample_weight[left].sum()
                loss += self._weighted_gini(y[right], sample_weight[right]) * sample_weight[right].sum()
                if loss < best_loss:
                    best_loss = float(loss)
                    self.feature_index_ = int(feature_index)
                    self.threshold_ = float(threshold)
                    self.left_probability_ = self._weighted_mean(y[left], sample_weight[left])
                    self.right_probability_ = self._weighted_mean(y[right], sample_weight[right])
        return self

    @staticmethod
    def _weighted_mean(y, weight):
        total = float(weight.sum())
        return float((y.astype(float) * weight).sum() / total) if total else 0.0

    @classmethod
    def _weighted_gini(cls, y, weight):
        p = cls._weighted_mean(y, weight)
        return 1.0 - p**2 - (1.0 - p) ** 2

    def predict_proba(self, X):
        X = _dense_array(X)
        probability = np.where(
            X[:, self.feature_index_] <= self.threshold_,
            self.left_probability_,
            self.right_probability_,
        )
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


class SimpleKNNClassifier(BaseEstimator, ClassifierMixin):
    """Dense Euclidean k-NN fallback for small coursework data."""

    def __init__(self, n_neighbors=15):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = _dense_array(X)
        self.y_ = np.asarray(y, dtype=bool)
        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X):
        X = _dense_array(X)
        k = min(int(self.n_neighbors), len(self.y_))
        probabilities = []
        for row in X:
            distances = ((self.X_ - row) ** 2).sum(axis=1)
            idx = np.argpartition(distances, k - 1)[:k]
            probabilities.append(float(self.y_[idx].mean()))
        probability = np.asarray(probabilities)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


class SimpleStumpEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Bootstrap ensemble fallback used only when RandomForest import is blocked."""

    def __init__(self, n_estimators=60, max_features="sqrt", class_weight=None, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X = _dense_array(X)
        y = np.asarray(y, dtype=bool)
        self.classes_ = np.array([False, True])
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            feature_count = max(1, int(np.sqrt(n_features)))
        else:
            feature_count = max(1, min(int(self.max_features), n_features))
        self.estimators_ = []
        for _ in range(int(self.n_estimators)):
            rows = rng.integers(0, len(y), len(y))
            cols = np.sort(rng.choice(n_features, size=feature_count, replace=False))
            stump = SimpleDecisionStumpClassifier(
                class_weight=self.class_weight,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(X[rows][:, cols], y[rows])
            self.estimators_.append((cols, stump))
        return self

    def predict_proba(self, X):
        X = _dense_array(X)
        probabilities = [
            estimator.predict_proba(X[:, cols])[:, 1]
            for cols, estimator in self.estimators_
        ]
        probability = np.mean(probabilities, axis=0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


def _optional_classifier(module_name, class_name, fallback, **kwargs):
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    except ImportError:
        return fallback


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": _optional_classifier(
            "sklearn.tree",
            "DecisionTreeClassifier",
            SimpleDecisionStumpClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
        "k-NN": _optional_classifier(
            "sklearn.neighbors",
            "KNeighborsClassifier",
            SimpleKNNClassifier(n_neighbors=15),
            n_neighbors=15,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": _optional_classifier(
            "sklearn.ensemble",
            "RandomForestClassifier",
            SimpleStumpEnsembleClassifier(
                n_estimators=60,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_estimators=200,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(classifier, numeric_features=None, categorical_features=None):
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", classifier),
        ]
    )


def _positive_probability(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(True)]


def metric_row(model_name, y_true, y_pred, y_prob=None):
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        row["roc_auc"] = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return row


def evaluate_baselines(y_true):
    return pd.DataFrame(
        [
            metric_row("Always on-time", y_true, np.zeros(len(y_true), dtype=bool)),
            metric_row("Always delayed", y_true, np.ones(len(y_true), dtype=bool)),
        ]
    )


def compare_models(
    train_df,
    eval_df,
    feature_columns=None,
    numeric_features=None,
    categorical_features=None,
):
    feature_columns = feature_columns or ACTIVE_FEATURE_COLUMNS
    numeric_features = numeric_features or ACTIVE_NUMERIC_FEATURES
    categorical_features = categorical_features or ACTIVE_CATEGORICAL_FEATURES
    validate_feature_contract(feature_columns)
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_eval = eval_df[feature_columns]
    y_eval = eval_df[TARGET_COLUMN]

    rows = []
    fitted = {}
    for name, classifier in build_models().items():
        model = build_pipeline(classifier, numeric_features, categorical_features)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        y_prob = _positive_probability(model, X_eval)
        rows.append(metric_row(name, y_eval, y_pred, y_prob))
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values(
        ["f2", "balanced_accuracy", "mcc"], ascending=False
    )
    return metrics, fitted


def compare_feature_sets(train_df, eval_df):
    feature_sets = [
        {
            "feature_set": "full_pre_dispatch",
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "note": "Includes engineered deterministic columns for coursework comparison.",
        },
        {
            "feature_set": "compact_nonredundant",
            "feature_columns": COMPACT_FEATURE_COLUMNS,
            "numeric_features": COMPACT_NUMERIC_FEATURES,
            "categorical_features": COMPACT_CATEGORICAL_FEATURES,
            "note": "Drops formula-derived columns and uses pizza_size_score encoding.",
        },
    ]
    rows = []
    for spec in feature_sets:
        validate_feature_contract(spec["feature_columns"])
        model = build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            spec["numeric_features"],
            spec["categorical_features"],
        )
        model.fit(train_df[spec["feature_columns"]], train_df[TARGET_COLUMN])
        y_pred = model.predict(eval_df[spec["feature_columns"]])
        y_prob = _positive_probability(model, eval_df[spec["feature_columns"]])
        row = metric_row(spec["feature_set"], eval_df[TARGET_COLUMN], y_pred, y_prob)
        row["feature_count"] = len(spec["feature_columns"])
        row["note"] = spec["note"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f2", "balanced_accuracy", "mcc"], ascending=False)


def select_best_model(dev_metrics):
    return dev_metrics.iloc[0]["model"]


def cross_validate_models(train_df, k=5):
    """Stratified k-fold CV on train to show stability (mean +/- std), not a
    single fragile dev split.
    """
    X = train_df[ACTIVE_FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    def _roc_auc_positive(estimator, X_fold, y_fold):
        # Robust for both sklearn and the pure-Python fallback estimators:
        # extract the positive-class probability explicitly instead of relying on
        # the built-in "roc_auc" scorer's class-detection (which fails on the
        # custom fallback classifiers).
        score = _positive_probability(estimator, X_fold)
        return roc_auc_score(np.asarray(y_fold, dtype=int), score)

    scoring = {
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": _roc_auc_positive,
    }
    rows = []
    for name, classifier in build_models().items():
        pipe = build_pipeline(classifier, ACTIVE_NUMERIC_FEATURES, ACTIVE_CATEGORICAL_FEATURES)
        result = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        row = {"model": name, "cv_folds": k}
        for metric in scoring:
            row[f"{metric}_mean"] = float(result[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(result[f"test_{metric}"].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f2_mean", ascending=False).reset_index(drop=True)


def tune_selected_model(train_df, k=5):
    """Dò siêu tham số cho mô hình tốt nhất (Logistic Regression) bằng GridSearchCV
    trên tập train với StratifiedKFold. Metric lựa chọn là F2.
    """
    X = train_df[ACTIVE_FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]
    
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scoring = make_scorer(fbeta_score, beta=2, zero_division=0)
    
    # Chỉ tune LogisticRegression vì nó đang là model khóa tốt nhất.
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    pipe = build_pipeline(classifier, ACTIVE_NUMERIC_FEATURES, ACTIVE_CATEGORICAL_FEATURES)
    
    param_grid = {
        "model__C": [0.1, 0.3, 1, 3, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        return_train_score=False,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    results = pd.DataFrame(grid_search.cv_results_)
    rows = []
    for i in range(len(results)):
        rows.append({
            "param_C": results.loc[i, "param_model__C"],
            "mean_test_f2": results.loc[i, "mean_test_score"],
            "std_test_f2": results.loc[i, "std_test_score"],
        })
    
    df_results = pd.DataFrame(rows).sort_values("mean_test_f2", ascending=False).reset_index(drop=True)
    df_results.to_csv(METRICS_DIR / "hyperparameter_tuning.csv", index=False)
    
    return df_results, grid_search.best_params_, grid_search.best_score_


def bootstrap_test_metrics(model, test_df, n_boot=2000, seed=RANDOM_STATE):
    """Percentile bootstrap CI for test metrics. Test has ~42 delayed rows, so
    point estimates are fragile; this quantifies that uncertainty.
    """
    rng = np.random.default_rng(seed)
    X = test_df[ACTIVE_FEATURE_COLUMNS].reset_index(drop=True)
    y = test_df[TARGET_COLUMN].to_numpy()
    pred = model.predict(X)
    n = len(y)
    samples = {"f2": [], "balanced_accuracy": [], "mcc": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y[idx], pred[idx]
        if len(set(yt)) < 2:
            continue
        samples["f2"].append(fbeta_score(yt, yp, beta=2, zero_division=0))
        samples["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        samples["mcc"].append(matthews_corrcoef(yt, yp))
        samples["recall"].append(recall_score(yt, yp, zero_division=0))
    point = {
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "recall": recall_score(y, pred, zero_division=0),
    }
    rows = []
    for metric, draws in samples.items():
        draws = np.asarray(draws)
        rows.append(
            {
                "metric": metric,
                "point_estimate": round(float(point[metric]), 4),
                "ci_low_2_5": round(float(np.percentile(draws, 2.5)), 4),
                "ci_high_97_5": round(float(np.percentile(draws, 97.5)), 4),
                "bootstrap_std": round(float(draws.std()), 4),
                "n_boot": int(len(draws)),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_figure(model, dev_df):
    """Precision-Recall curve on dev with the F2-optimal operating point marked."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X = dev_df[ACTIVE_FEATURE_COLUMNS]
    y = dev_df[TARGET_COLUMN].astype(int)
    prob = _positive_probability(model, X)
    precision, recall, thresholds = precision_recall_curve(y, prob)
    p, r = precision[:-1], recall[:-1]
    denom = 4 * p + r
    f2 = np.where(denom > 0, 5 * p * r / denom, 0)
    best = int(np.argmax(f2))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, marker=".", markersize=3)
    ax.scatter(r[best], p[best], color="red", zorder=5,
               label=f"F2-optimal @ thr={thresholds[best]:.2f} (F2={f2[best]:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve on dev (locked model)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "pr_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_model_figures(dev_metrics, fitted, dev_df, test_df, best_name, best_model):
    """Save evaluation charts so notebooks, slides and the report share one source.

    Returns the list of figure paths written.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Model comparison on dev across the decision-relevant metrics.
    compare = dev_metrics.set_index("model")[["f2", "balanced_accuracy", "mcc"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    compare.plot.bar(ax=ax)
    ax.set_title("Model comparison on dev (selection metrics)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # 2. Confusion matrix of the locked model on the test split.
    y_test = test_df[TARGET_COLUMN]
    test_pred = best_model.predict(test_df[ACTIVE_FEATURE_COLUMNS])
    cm = confusion_matrix(y_test, test_pred, labels=[False, True])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["On-time", "Delayed"])
    ax.set_yticks([0, 1], labels=["On-time", "Delayed"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix on test ({best_name})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIGURES_DIR / "confusion_matrix_test.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # 3. ROC curves on dev for every model that exposes probabilities.
    X_dev = dev_df[ACTIVE_FEATURE_COLUMNS]
    y_dev = dev_df[TARGET_COLUMN]
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        prob = _positive_probability(model, X_dev)
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_dev.astype(int), prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_dev.astype(int), prob):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on dev")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # 4. Linear-model coefficients of the locked model when available.
    model_step = best_model.named_steps.get("model")
    if model_step is not None and hasattr(model_step, "coef_"):
        try:
            names = best_model.named_steps["preprocess"].get_feature_names_out()
            coefs = pd.Series(model_step.coef_.ravel(), index=names)
            top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top.index, top.values, color=np.where(top.values >= 0, "#c0392b", "#2980b9"))
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"Top coefficients ({best_name}) — positive raises delay odds")
            ax.set_xlabel("Coefficient")
            fig.tight_layout()
            path = FIGURES_DIR / "model_coefficients.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
        except Exception:  # pragma: no cover - feature-name extraction is best effort.
            pass

    paths.append(build_threshold_figure(best_model, dev_df))
    return paths


def train_and_evaluate():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_processed_splits()
    baseline_dev = evaluate_baselines(dev_df[TARGET_COLUMN])
    baseline_test = evaluate_baselines(test_df[TARGET_COLUMN])
    feature_set_comparison = compare_feature_sets(train_df, dev_df)

    dev_metrics, fitted = compare_models(train_df, dev_df)
    best_name = select_best_model(dev_metrics)
    best_model = fitted[best_name]

    # Hyperparameter tuning for the selected model
    tuning_results = None
    best_params = None
    if best_name == "Logistic Regression":
        tuning_results, best_params, best_score = tune_selected_model(train_df, k=5)
        # Note: We keep the default model as the locked model if the tuning doesn't
        # drastically improve the result, as instructed ("tuning xác nhận default đã tốt").

    X_test = test_df[ACTIVE_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    test_pred = best_model.predict(X_test)
    test_prob = _positive_probability(best_model, X_test)
    test_metrics = pd.DataFrame([metric_row(best_name, y_test, test_pred, test_prob)])
    cv_metrics = cross_validate_models(train_df, k=5)
    bootstrap_ci = bootstrap_test_metrics(best_model, test_df, n_boot=2000)

    baseline_dev.round(4).to_csv(METRICS_DIR / "baseline_dev_metrics.csv", index=False)
    baseline_test.round(4).to_csv(METRICS_DIR / "baseline_test_metrics.csv", index=False)
    dev_metrics.round(4).to_csv(METRICS_DIR / "model_dev_comparison.csv", index=False)
    cv_metrics.round(4).to_csv(METRICS_DIR / "model_cv_metrics.csv", index=False)
    feature_set_comparison.round(4).to_csv(
        METRICS_DIR / "feature_set_comparison.csv", index=False
    )
    test_metrics.round(4).to_csv(METRICS_DIR / "model_test_metrics.csv", index=False)
    bootstrap_ci.round(4).to_csv(METRICS_DIR / "model_test_bootstrap_ci.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "best_delay_model.joblib")
    build_model_figures(dev_metrics, fitted, dev_df, test_df, best_name, best_model)

    summary = {
        "selection_metric": "f2",
        "active_feature_set": "compact_nonredundant",
        "active_feature_columns": ACTIVE_FEATURE_COLUMNS,
        "best_model": best_name,
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "test_rows": int(len(test_df)),
        "test_metrics": test_metrics.iloc[0].to_dict(),
        "cv_metrics_top": cv_metrics.iloc[0].to_dict(),
        "test_bootstrap_ci": bootstrap_ci.to_dict(orient="records"),
    }
    if best_params is not None:
        summary["tuning_best_params"] = best_params

    with open(METRICS_DIR / "model_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def load_best_model():
    path = MODELS_DIR / "best_delay_model.joblib"
    if not path.exists():
        train_and_evaluate()
    return joblib.load(path)


def predict_delay_probability(model, df):
    X = df[ACTIVE_FEATURE_COLUMNS]
    return _positive_probability(model, X)


if __name__ == "__main__":
    print(train_and_evaluate())
