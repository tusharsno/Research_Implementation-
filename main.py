"""
main.py
Complete diabetes risk prediction pipeline orchestrator.
Ordinal Regression + XAI (SHAP + LIME)
Clinical Progression: Normal(0) < Pre-Diabetic(1) < Diabetic(2)

Usage:
    python main.py --data path/to/dataset.csv --target Outcome --top_k 10
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from data_preprocessing import preprocess_pipeline, LABEL_MAP_INV
from feature_selection import select_top_features, plot_feature_importance
from model_training import (cross_validate_all, final_evaluation,
                             plot_benchmark_results, plot_confusion_matrix)
from explainability import run_xai_analysis
from ordinal_models import get_ordinal_models, get_baseline_models


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction Pipeline")
    parser.add_argument("--data",    type=str, default="diabetes_ipdd.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--target",  type=str, default="Risk_Level",
                        help="Target column name")
    parser.add_argument("--top_k",   type=int, default=10,
                        help="Number of top features to select")
    parser.add_argument("--cv_folds",type=int, default=5,
                        help="Cross-validation folds")
    parser.add_argument("--best_model", type=str, default="Ordinal_XGBoost",
                        help="Model for final eval and XAI")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save plots and results")
    parser.add_argument("--skip_xai", action="store_true",
                        help="Skip XAI analysis (faster run)")
    parser.add_argument("--skip_smote", action="store_true",
                        help="Skip Ordinal-SMOTE balancing")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  DIABETES RISK PREDICTION — ORDINAL REGRESSION + XAI")
    print("  Clinical Order: Normal(0) < Pre-Diabetic(1) < Diabetic(2)")
    print("="*65)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: '{args.data}'\n"
                                f"Please provide a valid CSV path via --data")

    print("\n── Step 1: Preprocessing ──")
    X, y, feature_names, scaler = preprocess_pipeline(
        args.data, target_col=args.target, apply_smote=not args.skip_smote
    )

    # ── Step 2: Feature Selection ─────────────────────────────────────────────
    print("\n── Step 2: Feature Selection (⚠️ Leakage Check Enabled) ──")
    X_sel, selected_features, ranking = select_top_features(
        X, y, feature_names, top_k=args.top_k, exclude_leakage=True
    )
    plot_feature_importance(ranking, save_path=f"{args.output_dir}/feature_importance.png")

    # ── Step 3: Train/Test Split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Split] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Step 4: Cross-Validation Benchmarking ─────────────────────────────────
    print(f"\n── Step 4: {args.cv_folds}-Fold Cross-Validation Benchmarking ──")
    summary_df = cross_validate_all(X_train, y_train, n_splits=args.cv_folds)

    print("\n\n[Results] Model Benchmark Summary:")
    # Print in professor's table format
    display_cols = ["Model", "Accuracy", "Macro_F1", "AUC_Macro", "G_Mean"]
    print(summary_df[display_cols].to_string(index=False))
    summary_df.to_csv(f"{args.output_dir}/benchmark_summary.csv", index=False)
    plot_benchmark_results(summary_df,
                           save_path=f"{args.output_dir}/benchmark_results.png")

    # ── Step 5: Final Evaluation ──────────────────────────────────────────────
    print(f"\n── Step 5: Final Evaluation — {args.best_model} ──")
    best_model, metrics = final_evaluation(
        X_train, y_train, X_test, y_test, model_name=args.best_model
    )
    plot_confusion_matrix(
        y_test, metrics["y_pred"], model_name=args.best_model,
        save_path=f"{args.output_dir}/confusion_matrix_{args.best_model}.png"
    )

    # ── Step 6: XAI Analysis ──────────────────────────────────────────────────
    if not args.skip_xai:
        print(f"\n── Step 6: XAI Analysis (SHAP + LIME) ──")
        run_xai_analysis(
            best_model, X_train, X_test, y_test,
            feature_names=selected_features,
            model_name=args.best_model,
            output_dir=args.output_dir
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Best Model : {args.best_model}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  AUC Macro  : {metrics['auc_macro']:.4f}")
    print(f"  MAE (Ord.) : {metrics['mae']:.4f}")
    print(f"  Outputs    : {os.path.abspath(args.output_dir)}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
