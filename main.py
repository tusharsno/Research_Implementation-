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
    parser.add_argument("--top_k",   type=int, default=14,
                        help="Number of top features to select")
    parser.add_argument("--cv_folds",type=int, default=5,
                        help="Cross-validation folds")
    parser.add_argument("--best_model", type=str, default="Ordinal_RandomForest",
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

    # ── Step 3: Train/Validation/Test Split (70/15/15) ───────────────────────
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sel, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: 50% of temp = 15% val, 50% of temp = 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"\n[Split] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"[Split] Ratio  → Train: {len(y_train)/len(y)*100:.1f}% | "
          f"Val: {len(y_val)/len(y)*100:.1f}% | "
          f"Test: {len(y_test)/len(y)*100:.1f}%")

    # ── Step 4: Cross-Validation Benchmarking (on train set) ─────────────────
    print(f"\n── Step 4: {args.cv_folds}-Fold Cross-Validation Benchmarking (Train Set) ──")
    summary_df = cross_validate_all(X_train, y_train, n_splits=args.cv_folds)

    print("\n\n[Results] Model Benchmark Summary (Professor's Table Format):")
    display_cols = ["Model", "Type", "Accuracy", "Macro_F1", "AUC_Macro", "G_Mean", "MAE"]
    print(summary_df[display_cols].to_string(index=False))
    print("\n[Note] Ordinal models respect N < P < Y ordering (MAE penalizes ordinal errors)")
    summary_df.to_csv(f"{args.output_dir}/benchmark_summary.csv", index=False)
    plot_benchmark_results(summary_df,
                           save_path=f"{args.output_dir}/benchmark_results.png")

    # ── Step 5: Validation Set Evaluation ───────────────────────────────────
    print(f"\n── Step 5: Validation Set Evaluation ──")
    print("[Validation] Evaluating best model on validation set (15%)...")
    _, val_metrics = final_evaluation(
        X_train, y_train, X_val, y_val, model_name=args.best_model
    )
    print(f"[Validation] Accuracy: {val_metrics['accuracy']:.4f} | "
          f"F1: {val_metrics['f1_macro']:.4f} | AUC: {val_metrics['auc_macro']:.4f}")

    # ── Step 6: Final Test Set Evaluation ─────────────────────────────────
    print(f"\n── Step 6: Final Test Set Evaluation — {args.best_model} ──")
    best_model, metrics = final_evaluation(
        X_train, y_train, X_test, y_test, model_name=args.best_model
    )
    plot_confusion_matrix(
        y_test, metrics["y_pred"], model_name=args.best_model,
        save_path=f"{args.output_dir}/confusion_matrix_{args.best_model}.png"
    )

    # ── Step 7: XAI Analysis ──────────────────────────────────────────────────
    if not args.skip_xai:
        print(f"\n── Step 7: XAI Analysis (SHAP + LIME) ──")
        run_xai_analysis(
            best_model, X_train, X_test, y_test,
            feature_names=selected_features,
            model_name=args.best_model,
            output_dir=args.output_dir
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Split          : 70% Train | 15% Validation | 15% Test")
    print(f"  Best Model     : {args.best_model}")
    print(f"  Val  Accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Test F1 Macro  : {metrics['f1_macro']:.4f}")
    print(f"  Test AUC Macro : {metrics['auc_macro']:.4f}")
    print(f"  Test MAE (Ord.): {metrics['mae']:.4f}")
    print(f"  Outputs        : {os.path.abspath(args.output_dir)}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
