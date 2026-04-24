# Technical Documentation
## Diabetes Risk Prediction System
### Ordinal Regression + Explainable AI (XAI)

---

## 1. System Overview

Predicts diabetes risk across three clinically ordered stages using ordinal regression and explainability frameworks.

**Clinical Progression:** Normal (0) → Pre-Diabetic (1) → Diabetic (2)

**Why Ordinal Regression?**
Standard multiclass treats all classes as independent. Diabetes has a natural progression — misclassifying Diabetic as Normal (2 levels off) is clinically worse than Pre-Diabetic (1 level off). Ordinal models penalize larger misclassifications via MAE.

---

## 2. Dataset

| Property | Value |
|---|---|
| File used by pipeline | `diabetes_ipdd.csv` |
| Source | `diabetes_12000_dataset_PERFECTED (1).csv` (Downloads) |
| Total Samples | 12,000 |
| Class Distribution | Normal: 518 (4.3%) \| Pre-Diabetic: 745 (6.2%) \| Diabetic: 10,737 (89.5%) |
| Target Column | `Risk_Level` (0=Normal, 1=Pre-Diabetic, 2=Diabetic) |

**Note:** Professor's original dataset is the Iraqi Patient Dataset (IPDD):
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05465-z

---

## 3. Project Structure

```
Research_Implementation/
├── main.py                    # Pipeline entry point
├── data_preprocessing.py      # Cleaning, encoding, Ordinal-SMOTE
├── feature_selection.py       # MI + Chi2 + RF ranking, leakage detection
├── ordinal_models.py          # OrdinalRandomForest, OrdinalXGBoost, mord
├── model_training.py          # CV, Accuracy/F1/AUC/G-Mean, benchmarking
├── explainability.py          # SHAP + LIME (consistency checked)
├── refine_dataset.py          # Data cleaning script
├── add_realistic_noise.py     # Noise injection script
├── test_setup.py              # Dependency verification
├── requirements.txt
├── diabetes_ipdd.csv          ← used by pipeline
├── diabetes_refined.csv       ← intermediate cleaned
├── diabetes_realistic.csv     ← noise-injected version
└── outputs/
    ├── benchmark_summary.csv
    ├── benchmark_results.png
    ├── confusion_matrix_Ordinal_XGBoost.png
    ├── feature_importance.png
    ├── shap_summary_Ordinal_XGBoost.png
    ├── shap_beeswarm_Ordinal_XGBoost.png
    ├── shap_local_Ordinal_XGBoost.png
    ├── lime_normal.png
    ├── lime_pre_diabetic.png
    └── lime_diabetic.png
```

---

## 4. Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Macro F1 | Unweighted mean F1 across all classes |
| AUC Macro | OvR AUC averaged across classes — strictly [0,1] |
| G-Mean | Geometric mean of per-class recall — balanced performance |
| MAE | Mean absolute ordinal error — penalizes larger class jumps |

**G-Mean formula:**
```
G-Mean = (Recall_Normal × Recall_PreDiabetic × Recall_Diabetic)^(1/3)
```

---

## 5. Feature Selection

**Leakage Features Excluded (diabetes consequences, not predictors):**
- `Diabetes_Duration`, `Nephropathy`, `Retinopathy`, `IHD`

**Top Features (IPDD dataset):**

| Rank | Feature | Score | Clinical Significance |
|---|---|---|---|
| 1 | HbA1c | 1.000 | Gold standard — 3-month glucose average |
| 2 | Cholesterol | 0.011 | Lipid metabolism |
| 3 | BMI | 0.008 | Obesity indicator |
| 4 | Medication | 0.007 | Treatment type |
| 5 | FBS | 0.004 | Fasting blood sugar |

---

## 6. Results

### Professor's Original Results

| Model | Accuracy | Macro F1 | AUC | G-Mean |
|---|---|---|---|---|
| OrdinalGradientBoosting | 1.000 | 1.000000 | 1.000000 | 1.000000 |
| RandomForest | 1.000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | 1.000 | 1.000000 | 1.000000 | 1.000000 |
| OrdinalRandomForest | 0.976 | 0.957074 | 0.985782 | 0.985782 |
| ProportionalOdds_LogisticIT | 0.943 | 0.903245 | 0.983184 | 0.953169 |

### Our Results (5-Fold CV on diabetes_ipdd.csv)

| Model | Accuracy | Macro F1 | AUC | G-Mean | Match |
|---|---|---|---|---|---|
| OrdinalGradientBoosting | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ Exact |
| RandomForest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ Exact |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ Exact |
| OrdinalRandomForest | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ⚠️ Higher |
| ProportionalOdds_LogisticIT | 0.980312 | 0.910728 | 0.998886 | 0.871829 | ~✅ Close |
| MLP | 0.996250 | 0.983738 | 0.999908 | 0.979797 | — |
| SVM | 0.982708 | 0.918564 | 0.998970 | 0.889088 | — |
| LogisticRegression | 0.968854 | 0.867411 | 0.996187 | 0.798790 | — |

**Note on differences:** Professor used a separate 1000-sample test set from "Dataset of Diabetes.csv". We use 80/20 split from the same dataset — this explains slight differences in OrdinalRandomForest and ProportionalOdds results.

---

## 7. Usage

```bash
# Run full pipeline
python3 main.py

# Skip XAI (faster)
python3 main.py --skip_xai

# Custom dataset
python3 main.py --data your_data.csv --target Risk_Level
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `diabetes_ipdd.csv` | Dataset CSV path |
| `--target` | `Risk_Level` | Target column |
| `--top_k` | 10 | Features to select |
| `--cv_folds` | 5 | CV folds |
| `--best_model` | `Ordinal_XGBoost` | Model for final eval + XAI |
| `--output_dir` | `outputs` | Output directory |
| `--skip_xai` | False | Skip SHAP + LIME |
| `--skip_smote` | False | Skip Ordinal-SMOTE |

---

## 8. Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | 1.8.0 | ML models, metrics |
| imbalanced-learn | 0.14.1 | SMOTE |
| mord | 0.7 | Proportional Odds |
| xgboost | 1.7.6 | Gradient boosting |
| shap | 0.51.0 | SHAP explainability |
| lime | 0.2.0.1 | LIME explainability |
| numpy | 2.4.4 | Numerical ops |
| pandas | 3.0.2 | Data manipulation |
| matplotlib | 3.10.9 | Plotting |
| seaborn | 0.13.2 | Visualization |

---

## 9. References

- Frank & Hall (2001). *A Simple Approach to Ordinal Classification*. ECML.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Ribeiro et al. (2016). *"Why Should I Trust You?"*. KDD.
- mord: https://github.com/fabianp/mord
- IPDD: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05465-z
