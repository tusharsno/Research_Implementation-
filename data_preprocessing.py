"""
data_preprocessing.py
Handles data cleaning, normalization, and class balancing for diabetes risk prediction.
Ordinal label encoding: Normal(0) < Pre-Diabetic(1) < Diabetic(2)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


# ── Label Mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {"N": 0, "P": 1, "Y": 2}
LABEL_MAP_INV = {0: "Normal", 1: "Pre-Diabetic", 2: "Diabetic"}

# Clinical features where zero is physiologically impossible
ZERO_INVALID_COLS = ["Cholesterol", "Creatinine", "BMI", "FBS", "HbA1c"]

# Columns to drop (redundant or non-predictive)
ADMIN_COLS = ["PatientID", "Name", "Date", "ID", "Risk"]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = ["Gender", "Ethnicity", "Medication"]

# Features to apply Min-Max scaling
SCALE_COLS = ["Age", "BMI", "FBS", "HbA1c", "Cholesterol",
              "Creatinine", "BP_Systolic", "BP_Diastolic",
              "Diabetes_Duration"]


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"[Load] Shape: {df.shape} | Columns: {list(df.columns)}")
    return df


def drop_admin_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-predictive administrative fields."""
    cols_to_drop = [c for c in ADMIN_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[Clean] Dropped admin columns: {cols_to_drop}")
    return df


def fix_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physiologically impossible zeros with column median.
    Applies only to clinically critical columns.
    """
    for col in ZERO_INVALID_COLS:
        if col not in df.columns:
            continue
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)
            print(f"[Clean] Fixed {zero_count} zero(s) in '{col}' → median={median_val:.3f}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns: Gender, Ethnicity, Medication.
    Drops first category to avoid multicollinearity.
    """
    cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_present:
        df = pd.get_dummies(df, columns=cols_present, drop_first=True, dtype=int)
        print(f"[Encode] One-hot encoded: {cols_present}")
    return df


def encode_labels(df: pd.DataFrame, target_col: str = "Risk_Level") -> pd.DataFrame:
    """
    Encode ordinal target: N→0, P→1, Y→2.
    Handles both string labels and existing numeric labels.
    """
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map(LABEL_MAP)
        unmapped = df[target_col].isna().sum()
        if unmapped > 0:
            raise ValueError(f"[Encode] {unmapped} unmapped label(s) found. "
                             f"Expected values: {list(LABEL_MAP.keys())}")
    df[target_col] = df[target_col].astype(int)
    print(f"[Encode] Label distribution: {dict(Counter(df[target_col]))}")
    return df


def normalize_features(df: pd.DataFrame, target_col: str = "Risk_Level",
                        scaler: MinMaxScaler = None):
    """
    Apply Min-Max scaling (0-1) to clinical features.
    Returns scaled DataFrame and fitted scaler.
    """
    feature_cols = [c for c in SCALE_COLS if c in df.columns]
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"[Normalize] Min-Max scaled {len(feature_cols)} features.")
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        print(f"[Normalize] Applied existing scaler to {len(feature_cols)} features.")
    return df, scaler


class OrdinalSMOTE:
    """
    Ordinal-aware SMOTE: applies SMOTE only between adjacent ordinal classes
    (N↔P and P↔Y) to preserve the clinical progression order.
    Specifically targets under-represented Pre-Diabetic class.
    """

    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        print(f"[OrdinalSMOTE] Before: {dict(Counter(y))}")

        X_res, y_res = X.copy(), y.copy()

        # Step 1: Balance N(0) ↔ P(1)
        mask_np = np.isin(y_res, [0, 1])
        X_np, y_np = X_res[mask_np], y_res[mask_np]
        if len(np.unique(y_np)) == 2:
            k = min(self.k_neighbors, Counter(y_np).most_common()[-1][1] - 1)
            if k >= 1:
                smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                X_np_res, y_np_res = smote.fit_resample(X_np, y_np)
                orig_p_count = (y_np == 1).sum()
                new_p_X = X_np_res[y_np_res == 1][orig_p_count:]
                new_p_y = y_np_res[y_np_res == 1][orig_p_count:]
                if len(new_p_X) > 0:
                    X_res = np.vstack([X_res, new_p_X])
                    y_res = np.concatenate([y_res, new_p_y])

        # Step 2: Balance P(1) ↔ Y(2)
        mask_py = np.isin(y_res, [1, 2])
        X_py, y_py = X_res[mask_py], y_res[mask_py]
        if len(np.unique(y_py)) == 2:
            k = min(self.k_neighbors, Counter(y_py).most_common()[-1][1] - 1)
            if k >= 1:
                smote = SMOTE(random_state=self.random_state, k_neighbors=k)
                X_py_res, y_py_res = smote.fit_resample(X_py, y_py)
                orig_p_count = (y_py == 1).sum()
                new_p_X = X_py_res[y_py_res == 1][orig_p_count:]
                new_p_y = y_py_res[y_py_res == 1][orig_p_count:]
                if len(new_p_X) > 0:
                    X_res = np.vstack([X_res, new_p_X])
                    y_res = np.concatenate([y_res, new_p_y])

        print(f"[OrdinalSMOTE] After:  {dict(Counter(y_res))}")
        return X_res, y_res


def preprocess_pipeline(filepath: str, target_col: str = "Risk_Level",
                         apply_smote: bool = True, scaler=None):
    """
    Full preprocessing pipeline.
    Returns: X_resampled, y_resampled, feature_names, fitted_scaler
    """
    df = load_data(filepath)
    df = drop_admin_columns(df)
    df = fix_zero_values(df)
    df = encode_labels(df, target_col)
    df = encode_categoricals(df)

    # Drop rows with remaining NaN
    before = len(df)
    df = df.dropna()
    print(f"[Clean] Dropped {before - len(df)} rows with NaN. Remaining: {len(df)}")

    df, scaler = normalize_features(df, target_col, scaler)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    if apply_smote:
        ordinal_smote = OrdinalSMOTE(random_state=42)
        X, y = ordinal_smote.fit_resample(X, y)

    print(f"[Pipeline] Final shape → X: {X.shape}, y: {y.shape}")
    return X, y, feature_cols, scaler
