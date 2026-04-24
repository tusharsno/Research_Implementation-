"""
model_training.py
Training pipeline, cross-validation, and benchmarking for all models.
Metrics: Accuracy, Macro F1, Macro AUC, G-Mean (matches professor's result table).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                              confusion_matrix, mean_absolute_error, accuracy_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

from ordinal_models import get_ordinal_models, get_baseline_models
from data_preprocessing import LABEL_MAP_INV

CLASSES = [0, 1, 2]
CLASS_NAMES = ["Normal", "Pre-Diabetic", "Diabetic"]

# Professor's exact model names from result table
DISPLAY_NAMES = {
    "Proportional_Odds":    "ProportionalOdds_LogisticIT",
    "Ordinal_RandomForest": "OrdinalRandomForest",
    "Ordinal_XGBoost":      "OrdinalGradientBoosting",
    "RandomForest":         "RandomForest",
    "XGBoost":              "XGBoost",
    "MLP":                  "MLP",
    "LogisticRegression":   "LogisticRegression",
    "SVM":                  "SVM",
}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax to ensure valid probability distribution [0,1]."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _safe_proba(model, X_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Get valid [0,1] probability matrix. Handles mord raw scores."""
    if not hasattr(model, "predict_proba"):
        return label_binarize(y_pred, classes=CLASSES).astype(float)
    raw = model.predict_proba(X_test)
    if raw.ndim == 2 and raw.shape[1] == 3:
        if (raw > 1.0).any() or (raw < 0).any():
            return _softmax(raw)
        clipped = np.clip(raw, 0, None)
        row_sums = clipped.sum(axis=1, keepdims=True)
        return clipped / np.where(row_sums == 0, 1, row_sums)
    return label_binarize(y_pred, classes=CLASSES).astype(float)


def _compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Macro OvR AUC, strictly validated in [0,1]."""
    try:
        y_bin = label_binarize(y_true, classes=CLASSES)
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        return float(np.clip(auc, 0.0, 1.0))
    except Exception as e:
        print(f"  [AUC Warning] {e}")
        return np.nan


def _compute_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Geometric Mean of per-class recall (sensitivity).
    G-Mean = (prod of per-class recall)^(1/K)
    Measures balanced performance across all classes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    per_class_recall = []
    for i in range(len(CLASSES)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            per_class_recall.append(cm[i, i] / row_sum)
    if not per_class_recall:
        return 0.0
    return float(np.prod(per_class_recall) ** (1.0 / len(per_class_recall)))


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit model and return all evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = _safe_proba(model, X_test, y_pred)

    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc_macro":   _compute_auc(y_test, y_proba),
        "gmean":       _compute_gmean(y_test, y_pred),
        "mae":         mean_absolute_error(y_test, y_pred),
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }


def cross_validate_all(X: np.ndarray, y: np.ndarray,
                        n_splits: int = 5) -> pd.DataFrame:
    """Stratified K-Fold CV for all ordinal + baseline models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_models = {**get_ordinal_models(), **get_baseline_models()}

    results = {name: {"accuracy": [], "f1_weighted": [], "f1_macro": [],
                       "auc_macro": [], "gmean": [], "mae": []}
               for name in all_models}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\n  Fold {fold}/{n_splits}", end="")

        for name, model in all_models.items():
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            for k in ["accuracy", "f1_weighted", "f1_macro", "auc_macro", "gmean", "mae"]:
                results[name][k].append(metrics[k])
            print(f" | {name}: Acc={metrics['accuracy']:.3f} F1={metrics['f1_macro']:.3f}", end="")

    summary = []
    for name, scores in results.items():
        summary.append({
            "Model":             DISPLAY_NAMES.get(name, name),
            "Type":              "Ordinal" if name in get_ordinal_models() else "Baseline",
            "Accuracy":          round(np.nanmean(scores["accuracy"]), 6),
            "Macro_F1":          round(np.nanmean(scores["f1_macro"]), 6),
            "AUC_Macro":         round(np.nanmean(scores["auc_macro"]), 6),
            "G_Mean":            round(np.nanmean(scores["gmean"]), 6),
            "F1_Weighted":       round(np.nanmean(scores["f1_weighted"]), 6),
            "MAE":               round(np.nanmean(scores["mae"]), 6),
        })

    df = pd.DataFrame(summary).sort_values("Accuracy", ascending=False)
    return df.reset_index(drop=True)


def final_evaluation(X_train, y_train, X_test, y_test,
                      model_name: str = "Ordinal_XGBoost"):
    """Train best model on full train set, evaluate on held-out test set."""
    all_models = {**get_ordinal_models(), **get_baseline_models()}
    model   = all_models[model_name]
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    display = DISPLAY_NAMES.get(model_name, model_name)

    print(f"\n[Final Eval] {display}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Macro F1    : {metrics['f1_macro']:.6f}")
    print(f"  AUC Macro   : {metrics['auc_macro']:.6f}")
    print(f"  G-Mean      : {metrics['gmean']:.6f}")
    print(f"  MAE (Ord.)  : {metrics['mae']:.4f}")
    print("\n" + classification_report(y_test, metrics["y_pred"],
                                       target_names=CLASS_NAMES, zero_division=0))
    return model, metrics


def plot_benchmark_results(summary_df: pd.DataFrame,
                            save_path: str = "benchmark_results.png"):
    """Comparison chart matching professor's result table format."""
    df = summary_df.copy()
    df["AUC_Macro"] = df["AUC_Macro"].clip(0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = {"Ordinal": "#e74c3c", "Baseline": "#3498db"}

    for ax, metric, label in zip(
        axes,
        ["Accuracy", "Macro_F1", "AUC_Macro"],
        ["Accuracy", "Macro F1", "Macro AUC"]
    ):
        bar_colors = [colors[t] for t in df["Type"]]
        bars = ax.barh(df["Model"][::-1], df[metric][::-1],
                       color=bar_colors[::-1], edgecolor="white", height=0.6)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, df[metric][::-1]):
            ax.text(min(bar.get_width(), 1.0) - 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{min(val,1.0):.3f}", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Ordinal Models"),
                       Patch(facecolor="#3498db", label="Baseline Models")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.suptitle("Diabetes Risk Prediction — Model Benchmarking\n"
                 "Clinical Order: Normal → Pre-Diabetic → Diabetic",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Benchmark results saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name,
                           save_path: str = "confusion_matrix.png"):
    """Percentage-annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    display = DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(f"Confusion Matrix (%) — {display}\n"
                 "Ordinal: Normal → Pre-Diabetic → Diabetic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")
