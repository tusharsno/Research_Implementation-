"""
explainability.py
XAI suite: SHAP (global + local) and LIME (local) for clinical interpretability.
Designed for the best-performing ordinal model (OrdinalXGBoost / OrdinalRandomForest).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import shap
import lime
import lime.lime_tabular

from data_preprocessing import LABEL_MAP_INV

CLASS_NAMES = ["Normal", "Pre-Diabetic", "Diabetic"]


# ── SHAP Analysis ──────────────────────────────────────────────────────────────

class SHAPAnalyzer:
    """
    SHAP-based global and local explanations.
    Uses TreeExplainer for tree-based models, KernelExplainer as fallback.
    """

    def __init__(self, model, X_train: np.ndarray, feature_names: list,
                 model_name: str = "Model"):
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self._build_explainer(X_train)

    def _build_explainer(self, X_train: np.ndarray):
        """Build appropriate SHAP explainer based on model type."""
        try:
            # TreeExplainer works for RF and XGBoost binary classifiers
            # For OrdinalXGBoost/OrdinalRF, we explain the underlying binary classifiers
            if hasattr(self.model, "binary_classifiers_"):
                # Use the last binary classifier (P vs Y — most clinically relevant)
                clf = self.model.binary_classifiers_[-1]
                self.explainer = shap.TreeExplainer(clf)
                self._explainer_type = "tree_binary"
                print(f"[SHAP] TreeExplainer on binary classifier (P→Y boundary)")
            elif hasattr(self.model, "estimators_"):
                self.explainer = shap.TreeExplainer(self.model)
                self._explainer_type = "tree_direct"
                print(f"[SHAP] TreeExplainer on direct model")
            else:
                # KernelExplainer for mord and other models
                background = shap.sample(X_train, min(100, len(X_train)))
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
                self._explainer_type = "kernel"
                print(f"[SHAP] KernelExplainer (slower, for non-tree model)")
        except Exception as e:
            print(f"[SHAP] Explainer build warning: {e}")
            background = shap.sample(X_train, min(50, len(X_train)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
            self._explainer_type = "kernel"

    def compute_shap_values(self, X: np.ndarray, max_samples: int = 200):
        """Compute SHAP values for a subset of samples."""
        X_sample = X[:min(max_samples, len(X))]
        try:
            self.shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            print(f"[SHAP] Warning during computation: {e}")
            self.shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
        self.X_sample = X_sample
        print(f"[SHAP] Computed values for {len(X_sample)} samples.")
        return self.shap_values

    def _get_sv_array(self):
        """Return a single 2D shap values array (use last binary classifier = P→Y)."""
        sv = self.shap_values
        if isinstance(sv, list):
            return sv[-1]   # Diabetic class / last binary boundary
        return sv

    def plot_summary(self, save_path: str = "shap_summary.png"):
        """Global SHAP summary bar plot — identifies key clinical factors."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        sv_plot = self._get_sv_array()
        shap.summary_plot(sv_plot, self.X_sample,
                          feature_names=self.feature_names,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Global Feature Importance — {self.model_name}\n"
                  "(P→Y Boundary: Pre-Diabetic vs Diabetic)", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Summary plot saved → {save_path}")

    def plot_beeswarm(self, save_path: str = "shap_beeswarm.png"):
        """SHAP beeswarm plot showing feature impact distribution."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        sv_plot = self._get_sv_array()
        shap.summary_plot(sv_plot, self.X_sample,
                          feature_names=self.feature_names,
                          show=False)
        plt.title(f"SHAP Beeswarm — {self.model_name}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Beeswarm plot saved → {save_path}")

    def plot_local_explanation(self, sample_idx: int,
                                save_path: str = "shap_local.png"):
        """Waterfall plot for a single patient's prediction explanation."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        sv_plot = self._get_sv_array()
        ev = self.explainer.expected_value
        expected = ev[-1] if isinstance(ev, (list, np.ndarray)) else ev

        explanation = shap.Explanation(
            values=sv_plot[sample_idx],
            base_values=float(expected),
            data=self.X_sample[sample_idx],
            feature_names=self.feature_names
        )
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"SHAP Local Explanation — Patient #{sample_idx}\n"
                  f"(Diabetic Risk Factors)", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Local explanation saved → {save_path}")

    def get_global_importance(self) -> pd.Series:
        """Return mean |SHAP| values as global feature importance."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")
        sv = self._get_sv_array()
        mean_abs = np.abs(sv).mean(axis=0)
        return pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)


# ── LIME Analysis ──────────────────────────────────────────────────────────────

class LIMEAnalyzer:
    """
    LIME-based local explanations for individual patient predictions.
    Provides clinically interpretable feature contributions per prediction.
    """

    def __init__(self, model, X_train: np.ndarray, feature_names: list,
                 model_name: str = "Model"):
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=CLASS_NAMES,
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        print(f"[LIME] Explainer initialized for {model_name}")

    def explain_instance(self, x: np.ndarray, num_features: int = 10,
                          num_samples: int = 1000) -> lime.lime_tabular.TableDomainMapper:
        """Generate LIME explanation for a single patient instance."""
        predict_fn = (self.model.predict_proba
                      if hasattr(self.model, "predict_proba")
                      else lambda X: np.eye(3)[self.model.predict(X)])
        exp = self.explainer.explain_instance(
            x, predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[0, 1, 2]
        )
        return exp

    def plot_local_explanation(self, x: np.ndarray, sample_idx: int,
                                true_label: int = None,
                                save_path: str = "lime_local.png"):
        """Bar chart of LIME feature contributions for a single patient."""
        exp = self.explain_instance(x)

        # Predicted class = argmax of predict_proba (consistency check)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x.reshape(1, -1))[0]
        else:
            proba = np.eye(3)[self.model.predict(x.reshape(1, -1))[0]]
        pred_class = int(np.argmax(proba))

        # Verify LIME predicted class matches model predicted class
        lime_pred = int(np.argmax(exp.predict_proba))
        if lime_pred != pred_class:
            print(f"  [LIME] Note: LIME local approx predicts {CLASS_NAMES[lime_pred]}, "
                  f"model predicts {CLASS_NAMES[pred_class]} — using model prediction")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for class_idx, (ax, class_name) in enumerate(zip(axes, CLASS_NAMES)):
            contributions = dict(exp.as_list(label=class_idx))
            # Sort by absolute contribution
            sorted_items = sorted(contributions.items(),
                                  key=lambda x: abs(x[1]), reverse=True)[:8]
            features = [item[0] for item in sorted_items]
            values   = [item[1] for item in sorted_items]
            colors   = ["#e74c3c" if v > 0 else "#3498db" for v in values]
            ax.barh(features[::-1], values[::-1], color=colors[::-1])
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_title(f"LIME: {class_name}\nP={proba[class_idx]:.3f}",
                         fontweight="bold" if class_idx == pred_class else "normal")
            ax.set_xlabel("Feature Contribution")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Highlight predicted class
            if class_idx == pred_class:
                ax.set_facecolor("#f8f9fa")

        true_str = f" | True: {CLASS_NAMES[true_label]}" if true_label is not None else ""
        match_str = " ✓" if true_label == pred_class else " ✗"
        fig.suptitle(f"LIME Local Explanation — Patient #{sample_idx} "
                     f"| Predicted: {CLASS_NAMES[pred_class]}{true_str}{match_str if true_label is not None else ''}\n"
                     f"Model: {self.model_name}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[LIME] Local explanation saved → {save_path} "
              f"(Predicted: {CLASS_NAMES[pred_class]}"
              f"{', Correct ✓' if true_label == pred_class else ', Wrong ✗' if true_label is not None else ''})")

    def explain_multiple(self, X: np.ndarray, y_true: np.ndarray,
                          n_samples: int = 3, save_dir: str = "."):
        """Generate LIME explanations for n representative patients."""
        # Pick one from each class
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_indices = np.where(y_true == class_idx)[0]
            if len(class_indices) == 0:
                continue
            idx = class_indices[0]
            save_path = f"{save_dir}/lime_{class_name.lower().replace('-', '_')}.png"
            self.plot_local_explanation(
                X[idx], sample_idx=idx,
                true_label=class_idx,
                save_path=save_path
            )


# ── Combined XAI Report ────────────────────────────────────────────────────────

def run_xai_analysis(model, X_train: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray, feature_names: list,
                      model_name: str = "Ordinal_XGBoost",
                      output_dir: str = "."):
    """
    Full XAI pipeline: SHAP global + local, LIME local explanations.
    """
    print(f"\n{'='*60}")
    print(f"XAI Analysis — {model_name}")
    print(f"{'='*60}")

    # ── SHAP ──
    shap_analyzer = SHAPAnalyzer(model, X_train, feature_names, model_name)
    shap_analyzer.compute_shap_values(X_test, max_samples=150)
    shap_analyzer.plot_summary(f"{output_dir}/shap_summary_{model_name}.png")
    shap_analyzer.plot_beeswarm(f"{output_dir}/shap_beeswarm_{model_name}.png")
    shap_analyzer.plot_local_explanation(0, f"{output_dir}/shap_local_{model_name}.png")

    global_imp = shap_analyzer.get_global_importance()
    print(f"\n[SHAP] Top 5 Global Features:\n{global_imp.head(5).to_string()}")

    # ── LIME ──
    lime_analyzer = LIMEAnalyzer(model, X_train, feature_names, model_name)
    lime_analyzer.explain_multiple(X_test, y_test, save_dir=output_dir)

    return shap_analyzer, lime_analyzer
