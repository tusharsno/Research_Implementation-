"""
refine_dataset.py
Cleans and refines the raw diabetes dataset before pipeline use.

Issues fixed:
  1. Medication NaN (366 rows) → filled with "None"
  2. Diabetes_Duration > 0 for Normal (Risk_Level=0) patients → set to 0
  3. Outliers in continuous features → IQR-based winsorization (capping)
  4. Risk column dropped (6488 mismatches with Risk_Level — redundant)
  5. Saves refined CSV for pipeline use
"""

import pandas as pd
import numpy as np

RAW_FILE    = "diabetes_balanced_dataset (1).csv"
OUTPUT_FILE = "diabetes_refined.csv"

NUMERIC_COLS = [
    "Age", "BMI", "HbA1c", "FBS", "Cholesterol",
    "Creatinine", "BP_Systolic", "BP_Diastolic", "Diabetes_Duration"
]


def fix_medication_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Medication with 'None' (patient on no medication)."""
    n = df["Medication"].isna().sum()
    df["Medication"] = df["Medication"].fillna("None")
    print(f"[Fix] Medication NaN: filled {n} rows with 'None'")
    return df


def fix_diabetes_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normal patients (Risk_Level=0) cannot have Diabetes_Duration > 0.
    Set to 0 for clinical consistency.
    """
    mask = (df["Risk_Level"] == 0) & (df["Diabetes_Duration"] > 0)
    n = mask.sum()
    df.loc[mask, "Diabetes_Duration"] = 0
    print(f"[Fix] Diabetes_Duration: reset {n} Normal patients to 0")
    return df


def winsorize_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based winsorization: cap values at [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Preserves distribution shape while removing extreme outliers.
    """
    total_capped = 0
    for col in NUMERIC_COLS:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        if n_lo + n_hi > 0:
            print(f"  {col}: capped {n_lo} low + {n_hi} high = {n_lo+n_hi} values "
                  f"→ range now [{lo:.2f}, {hi:.2f}]")
            total_capped += n_lo + n_hi
    print(f"[Fix] Winsorization: total {total_capped} values capped")
    return df


def report_before_after(before: pd.DataFrame, after: pd.DataFrame):
    """Print a quick comparison of key stats."""
    print("\n=== Before vs After Refinement ===")
    print(f"  Shape        : {before.shape} → {after.shape}")
    print(f"  Missing vals : {before.isnull().sum().sum()} → {after.isnull().sum().sum()}")
    print(f"  Class dist   : {dict(before['Risk_Level'].value_counts().sort_index())} "
          f"→ {dict(after['Risk_Level'].value_counts().sort_index())}")
    print(f"\n  Numeric ranges after refinement:")
    for col in NUMERIC_COLS:
        print(f"    {col:20s}: [{after[col].min():.2f}, {after[col].max():.2f}]")


def refine():
    print(f"[Load] Reading {RAW_FILE}...")
    df_raw = pd.read_csv(RAW_FILE)
    df = df_raw.copy()
    print(f"[Load] Shape: {df.shape}")

    # 1. Drop redundant Risk column (6488 mismatches with Risk_Level)
    if "Risk" in df.columns:
        df = df.drop(columns=["Risk"])
        print("[Fix] Dropped redundant 'Risk' column")

    # 2. Fix Medication NaN
    df = fix_medication_nan(df)

    # 3. Fix clinical inconsistency: Normal + Diabetes_Duration > 0
    df = fix_diabetes_duration(df)

    # 4. Winsorize outliers
    print("[Fix] Winsorizing outliers...")
    df = winsorize_outliers(df)

    # 5. Report
    report_before_after(df_raw, df)

    # 6. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Save] Refined dataset saved → {OUTPUT_FILE}")
    print(f"       Rows: {len(df)} | Columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    refine()
