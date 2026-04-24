"""
ordinal_models.py
Implements:
  1. Custom Ordinal Random Forest (sklearn-compatible wrapper)
  2. Proportional Odds (mord.LogisticAT)
  3. Ordinal XGBoost variant
  4. Standard multiclass baselines
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import mord
import xgboost as xgb


# ── 1. Custom Ordinal Random Forest ───────────────────────────────────────────

class OrdinalRandomForest(BaseEstimator, ClassifierMixin):
    """
    Ordinal Random Forest using Frank & Hall (2001) decomposition.
    Decomposes K-class ordinal problem into K-1 binary problems:
      - Binary 1: P(Y > 0)  →  Normal vs {Pre-Diabetic, Diabetic}
      - Binary 2: P(Y > 1)  →  {Normal, Pre-Diabetic} vs Diabetic
    Final class = argmax of reconstructed ordinal probabilities.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = None,
                 min_samples_split: int = 2, random_state: int = 42,
                 n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.binary_classifiers_ = []

        # Train K-1 binary classifiers
        for k in range(self.n_classes_ - 1):
            # Binary target: 1 if y > k, else 0
            y_binary = (y > k).astype(int)
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            clf.fit(X, y_binary)
            self.binary_classifiers_.append(clf)

        # Store feature importances as average across binary classifiers
        self.feature_importances_ = np.mean(
            [clf.feature_importances_ for clf in self.binary_classifiers_], axis=0
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # P(Y > k) for each binary classifier
        cumulative_probs = np.column_stack([
            clf.predict_proba(X)[:, 1]
            for clf in self.binary_classifiers_
        ])

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        # P(Y=0) = 1 - P(Y>0)
        proba[:, 0] = 1.0 - cumulative_probs[:, 0]

        # P(Y=k) = P(Y>k-1) - P(Y>k)  for 0 < k < K-1
        for k in range(1, self.n_classes_ - 1):
            proba[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]

        # P(Y=K-1) = P(Y>K-2)
        proba[:, -1] = cumulative_probs[:, -1]

        # Clip and renormalize to handle floating point issues
        proba = np.clip(proba, 0, 1)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ── 2. Ordinal XGBoost Variant ─────────────────────────────────────────────────

class OrdinalXGBoost(BaseEstimator, ClassifierMixin):
    """
    Ordinal XGBoost using the same Frank & Hall decomposition as OrdinalRandomForest
    but with XGBoost binary classifiers for stronger gradient boosting.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 random_state: int = 42, use_label_encoder: bool = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.use_label_encoder = use_label_encoder

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.binary_classifiers_ = []

        for k in range(self.n_classes_ - 1):
            y_binary = (y > k).astype(int)
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0
            )
            clf.fit(X, y_binary)
            self.binary_classifiers_.append(clf)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        cumulative_probs = np.column_stack([
            clf.predict_proba(X)[:, 1]
            for clf in self.binary_classifiers_
        ])

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        proba[:, 0] = 1.0 - cumulative_probs[:, 0]
        for k in range(1, self.n_classes_ - 1):
            proba[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
        proba[:, -1] = cumulative_probs[:, -1]

        proba = np.clip(proba, 0, 1)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ── 3. Model Registry ──────────────────────────────────────────────────────────

def get_ordinal_models() -> dict:
    """Returns the three ordinal models for comparison."""
    return {
        "Proportional_Odds": mord.LogisticAT(alpha=1.0, max_iter=1000),
        "Ordinal_RandomForest": OrdinalRandomForest(n_estimators=200, random_state=42),
        "Ordinal_XGBoost": OrdinalXGBoost(n_estimators=200, learning_rate=0.1,
                                           random_state=42),
    }


def get_baseline_models() -> dict:
    """Returns standard multiclass classifiers as benchmarks."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    import xgboost as xgb

    return {
        "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42,
                                                      n_jobs=-1),
        "XGBoost":            xgb.XGBClassifier(n_estimators=200, random_state=42,
                                                 use_label_encoder=False,
                                                 eval_metric="mlogloss", verbosity=0),
        "MLP":                MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                            random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                SVC(kernel="rbf", probability=True, random_state=42),
    }
