"""
feature_selection.py
Multi-method feature selection: Mutual Information, Chi-Square, Random Forest importance.
Highlights HbA1c, FBS, BMI and lipid metabolism as key biological levers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier

# Clinically expected important features — always highlighted in plots
CLINICAL_LEVERS = ["HbA1c", "FBS", "BMI", "Cholesterol", "Glucose"]

# Features that may cause data leakage (directly derived from target)
LEAKAGE_RISK_COLS = ["Diabetes_Duration", "Nephropathy", "Retinopathy", "IHD"]


def compute_mutual_information(X: np.ndarray, y: np.ndarray,
                                feature_names: list) -> pd.Series:
    """Mutual Information scores between each feature and the ordinal target."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)


def compute_chi_square(X: np.ndarray, y: np.ndarray,
                        feature_names: list) -> pd.Series:
    """Chi-Square scores. Requires non-negative values (Min-Max scaled data is fine)."""
    X_clipped = np.clip(X, 0, None)
    chi_scores, _ = chi2(X_clipped, y)
    return pd.Series(chi_scores, index=feature_names).sort_values(ascending=False)


def compute_rf_importance(X: np.ndarray, y: np.ndarray,
                           feature_names: list) -> pd.Series:
    """Random Forest feature importance (Gini impurity based)."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_,
                     index=feature_names).sort_values(ascending=False)


def check_leakage(feature_names: list) -> list:
    """
    Warn about features that may cause data leakage.
    Diabetes_Duration, Nephropathy, Retinopathy, IHD are consequences
    of diabetes — not causes. Including them inflates model performance.
    """
    leaky = [f for f in LEAKAGE_RISK_COLS if f in feature_names]
    if leaky:
        print(f"[Leakage Warning] These features are diabetes consequences, "
              f"not predictors: {leaky}")
        print(f"  Consider excluding them for causal prediction.")
    return leaky


def rank_features(X: np.ndarray, y: np.ndarray,
                   feature_names: list) -> pd.DataFrame:
    """
    Combine MI + Chi2 + RF into unified ranking.
    Each method normalized to [0,1] then averaged.
    """
    # Leakage check
    check_leakage(feature_names)

    mi  = compute_mutual_information(X, y, feature_names)
    chi = compute_chi_square(X, y, feature_names)
    rf  = compute_rf_importance(X, y, feature_names)

    def normalize(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    df = pd.DataFrame({
        "MutualInfo":    normalize(mi),
        "ChiSquare":     normalize(chi),
        "RF_Importance": normalize(rf),
    })
    df["Combined_Score"] = df.mean(axis=1)
    df = df.sort_values("Combined_Score", ascending=False)

    print("\n[Feature Selection] Top 10 Features:")
    print(df.head(10).to_string())
    return df


def select_top_features(X: np.ndarray, y: np.ndarray,
                         feature_names: list, top_k: int = 10,
                         exclude_leakage: bool = True):
    """
    Select top-k features using combined ranking.
    - Optionally excludes leakage-risk features (Diabetes_Duration etc.)
    - Always includes clinical levers (HbA1c, FBS, BMI, Cholesterol)
    Returns: X_selected, selected_feature_names, full_ranking_df
    """
    # Optionally remove leakage features before ranking
    if exclude_leakage:
        leaky = [f for f in LEAKAGE_RISK_COLS if f in feature_names]
        if leaky:
            keep_idx = [i for i, f in enumerate(feature_names) if f not in leaky]
            X_clean = X[:, keep_idx]
            clean_names = [feature_names[i] for i in keep_idx]
            print(f"[Feature Selection] Excluded leakage features: {leaky}")
        else:
            X_clean, clean_names = X, feature_names
    else:
        X_clean, clean_names = X, feature_names

    ranking = rank_features(X_clean, y, clean_names)
    top_features = ranking.head(top_k).index.tolist()

    # Force-include clinical levers if present
    for pf in CLINICAL_LEVERS:
        if pf in clean_names and pf not in top_features:
            top_features.append(pf)
            print(f"[Feature Selection] Force-included clinical lever: {pf}")

    indices = [clean_names.index(f) for f in top_features]
    X_selected = X_clean[:, indices]
    print(f"[Feature Selection] Final {len(top_features)} features: {top_features}")
    return X_selected, top_features, ranking


def plot_feature_importance(ranking: pd.DataFrame, top_k: int = 15,
                             save_path: str = "feature_importance.png"):
    """Horizontal bar chart — clinical levers highlighted in red."""
    top = ranking.head(top_k).copy()
    colors = ["#e74c3c" if f in CLINICAL_LEVERS else "#3498db" for f in top.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top.index[::-1], top["Combined_Score"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, top["Combined_Score"][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Normalized Combined Score (MI + Chi² + RF)", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_title(f"Top {top_k} Features — Multi-Method Importance\n"
                 "Red = Clinical Biological Levers (HbA1c, FBS, BMI, Cholesterol)",
                 fontsize=11)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#e74c3c", label="Clinical Lever"),
              Patch(facecolor="#3498db", label="Other Feature")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Feature importance saved → {save_path}")
