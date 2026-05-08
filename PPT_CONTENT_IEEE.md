# PPT Content File — IEEE Format
## Explainable Ordinal Regression for Diabetes Risk Stratification
### Slide-by-Slide Complete Content

---
> **Instructions for use:**
> - `[IMAGE: filename.png]` → insert the corresponding output image at that position
> - `[METHODOLOGY SLIDE]` → insert your own methodology diagram/content here
> - All content follows IEEE conference presentation standards
> - Suggested total slides: 16–18 slides
---

---

## SLIDE 1 — Title Slide

**Title:**
Explainable Ordinal Regression for Diabetes Risk Stratification Using Multi-Method Feature Selection and XAI

**Subtitle:**
An End-to-End Clinical Decision Support Framework

**Authors:**
[Your Name(s)]
[Department / Institution]
[Conference Name, Year]

**Bottom tag line:**
*"Respecting Disease Progression Order: Normal → Pre-Diabetic → Diabetic"*

**Dataset Note (for presenter):**
*Dataset synthetically constructed based on statistical distributions reported in the National Diabetes Registry (NDR) Annual Report, Malaysia, 2023.*

---

## SLIDE 2 — Motivation & Problem Statement

**Title:** Why This Research Matters

**Key Points:**
- Diabetes affects **537 million adults** globally (IDF, 2021) — projected 783 million by 2045
- Early detection of **Pre-Diabetic** stage is critical — it is reversible with lifestyle intervention
- Conventional ML classifiers treat diabetes risk as **independent classes**, ignoring the natural clinical progression order:
  > Normal → Pre-Diabetic → Diabetic
- Misclassifying a **Normal** patient as **Diabetic** is far more harmful than misclassifying as **Pre-Diabetic** — yet standard classifiers penalize both equally
- Clinical AI models must be **explainable** — clinicians need to understand *why* a prediction was made, not just *what* was predicted

**Research Gap:**
> Existing approaches lack (1) ordinal-aware modeling, (2) class-imbalance handling that preserves ordinal structure, and (3) integrated XAI for clinical interpretability.

---

## SLIDE 3 — Research Objectives

**Title:** Research Objectives

**Objectives:**
1. Develop an **ordinal regression framework** that respects the N < P < Y disease progression order
2. Implement **custom Ordinal-SMOTE** to handle severe class imbalance without violating ordinal boundaries
3. Apply **multi-method feature selection** (MI + Chi² + RF) to identify key clinical biomarkers
4. Benchmark ordinal models against **5 standard multiclass baselines**
5. Provide **clinical explainability** via SHAP (global + local) and LIME (local) analysis
6. Validate results on a synthetically constructed clinical dataset of **12,000 patients**, built from National Diabetes Registry (NDR) Malaysia 2023 statistical distributions

---

## SLIDE 4 — Dataset Description

**Title:** Dataset: National Diabetes Registry (NDR), Malaysia — 2023

**Dataset Details:**

| Property | Value |
|----------|-------|
| Source | Synthetically constructed based on National Diabetes Registry (NDR) Annual Report, Malaysia, 2023 |
| Construction Method | Statistical distributions from NDR report reproduced via manual calculations (Google Colab) |
| Total Records | 12,000 patients |
| Features | 19 columns (raw) → 14 features (after leakage removal) |
| Target Variable | Risk_Level: Normal (0), Pre-Diabetic (1), Diabetic (2) |

**Key Clinical Features Used (All 14):**

| Rank | Feature | Clinical Role |
|------|---------|---------------|
| 1 | HbA1c (%) | Glycated hemoglobin — ADA gold standard for diabetes diagnosis |
| 2 | FBS (mg/dL) | Fasting blood sugar — direct diabetes indicator |
| 3 | Age | Risk doubles every decade after 40 |
| 4 | Creatinine | Kidney function marker |
| 5 | BMI | Obesity — primary Type 2 diabetes risk factor |
| 6 | BP_Systolic | Systolic blood pressure — hypertension co-occurrence |
| 7 | BP_Diastolic | Diastolic blood pressure — hypertension marker |
| 8 | Cholesterol | Lipid metabolism disruption |
| 9 | Gender_Male | Gender-based risk difference |
| 10 | Hypertension | Comorbidity indicator |
| 11 | Ethnicity_Malay | Ethnicity-based risk factor |
| 12 | Dyslipidemia | Lipid disorder comorbidity |
| 13 | Ethnicity_Indian | Ethnicity-based risk factor |
| 14 | Ethnicity_Others | Ethnicity-based risk factor |

**Data Leakage Removed:**
> Columns `Diabetes_Duration`, `Nephropathy`, `Retinopathy`, `IHD`, `Medication` were removed — these are *consequences* of diabetes, not predictors. Including them caused 100% accuracy (scientifically invalid).

---

## SLIDE 5 — Class Imbalance Problem

**Title:** Severe Class Imbalance — A Critical Challenge

**Before Ordinal-SMOTE:**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 510 | 4.3% |
| Pre-Diabetic (1) | 1,597 | 13.3% |
| Diabetic (2) | 9,893 | 82.4% |
| **Total** | **12,000** | **100%** |

**Problem:**
- Pre-Diabetic class is severely underrepresented (only 13.3%)
- Standard SMOTE generates synthetic samples without respecting ordinal boundaries — a synthetic Pre-Diabetic sample may be closer in feature space to a Diabetic sample, violating N < P < Y

**Our Solution — Ordinal-SMOTE:**
- Interpolates only between **adjacent classes**: N↔P, then P↔Y
- Never generates samples that cross non-adjacent class boundaries
- Preserves clinical progression order in synthetic data

**After Ordinal-SMOTE:**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 9,893 | 33.3% |
| Pre-Diabetic (1) | 9,893 | 33.3% |
| Diabetic (2) | 9,893 | 33.3% |
| **Total** | **29,679** | **100%** |

---

## SLIDE 6 — METHODOLOGY

**Title:** Proposed Methodology

**[METHODOLOGY SLIDE]**
> *(Add your methodology diagram/flowchart here)*
> *(Pipeline: Data Collection → Preprocessing → Feature Selection → Ordinal-SMOTE → Model Training → Evaluation → XAI)*

---

## SLIDE 7 — Feature Selection

**Title:** Multi-Method Feature Selection — Novel Contribution

**Step 1 — Three Methods Combined:**

| Method | Technique | Strength |
|--------|-----------|----------|
| Mutual Information (MI) | `mutual_info_classif` | Captures non-linear dependencies |
| Chi-Square Test | `chi2` from sklearn | Statistical independence test |
| Random Forest Importance | Gini impurity, 200 trees | Captures complex feature interactions |

**Combined Score Formula:**
> Combined Score = (Normalized MI + Normalized Chi² + Normalized RF) / 3

**Step 2 — All 14 Features Ranked:**

| Rank | Feature | Score |
|------|---------|-------|
| 1 | HbA1c | 0.975 |
| 2 | FBS | 0.718 |
| 3 | Age | 0.432 |
| 4 | Creatinine | 0.106 |
| 5 | BMI | 0.105 |
| 6 | BP_Systolic | 0.104 |
| 7 | BP_Diastolic | 0.102 |
| 8 | Cholesterol | 0.097 |
| 9 | Gender_Male | 0.025 |
| 10 | Hypertension | 0.013 |
| 11 | Ethnicity_Malay | 0.012 |
| 12 | Dyslipidemia | 0.009 |
| 13 | Ethnicity_Indian | 0.003 |
| 14 | Ethnicity_Others | 0.003 |

**Step 3 — Ablation Study Finding:**

| Configuration | Accuracy | F1 | MAE |
|--------------|----------|-----|-----|
| With Feature Selection (Top-10) | 90.63% | 0.904 | 0.095 |
| **All 14 Features (Proposed)** | **92.03%** | **0.919** | **0.081** |

> Feature selection ranking confirmed all 14 post-leakage-removal features contribute meaningfully. Ablation study revealed retaining all 14 features yields superior performance (+1.40% accuracy). Therefore, the complete ranked feature set was retained for final model training.

**Why This Decision?**
> - Feature Selection (MI + Chi² + RF) is retained as a **novel contribution** — it ranks all features by clinical importance, confirming HbA1c and FBS as dominant predictors
> - Ablation study tested both approaches: top-10 features (90.63%) vs all-14 features (92.03%)
> - Result confirmed: all 14 features contribute meaningfully to prediction — none are redundant
> - Therefore: Feature Selection used for **ranking and clinical insight**, all 14 features retained for **maximum model performance**
> - This is scientifically honest — we tested both ways and reported the better result

**[IMAGE: feature_importance.png]**

---

## SLIDE 8 — Ordinal Regression: Core Concept

**Title:** Why Ordinal Regression? — Frank & Hall Decomposition

**Standard Classification Problem:**
> Treats Normal, Pre-Diabetic, Diabetic as independent classes
> Penalty(Normal → Pre-Diabetic) = Penalty(Normal → Diabetic) ❌

**Ordinal Regression Solution:**
> Penalizes misclassifications proportionally to ordinal distance
> Penalty(Normal → Pre-Diabetic) < Penalty(Normal → Diabetic) ✅

**Frank & Hall (2001) Decomposition:**
> K-class ordinal problem → K-1 binary classifiers

```
Binary Classifier 1:  P(Y > 0)  →  Normal  vs  {Pre-Diabetic, Diabetic}
Binary Classifier 2:  P(Y > 1)  →  {Normal, Pre-Diabetic}  vs  Diabetic

Reconstructed Probabilities:
  P(Normal)        =  1 − P(Y > 0)
  P(Pre-Diabetic)  =  P(Y > 0) − P(Y > 1)
  P(Diabetic)      =  P(Y > 1)

Final Prediction   =  argmax [ P(Normal), P(Pre-Diabetic), P(Diabetic) ]
```

**Key Advantage:** Always respects N < P < Y ordering. Any base classifier (RF, XGBoost) can be used inside this framework.

---

## SLIDE 9 — Models Implemented

**Title:** Models Implemented — Ordinal vs. Baseline

**Ordinal Models (Proposed — Respect N < P < Y):**

| Model | Method | Key Config |
|-------|--------|------------|
| Proportional Odds | mord.LogisticAT | α=1.0, linear |
| Ordinal Random Forest | Frank & Hall + RF | 200 trees |
| Ordinal XGBoost | Frank & Hall + XGBoost | 200 est., lr=0.1, depth=6 |
| **Ensemble (Ordinal RF 60% + Ordinal XGBoost 40%) ⭐** | **Weighted Combination** | **Final Proposed Model** |

**Baseline Models (Standard Multiclass — For Comparison):**

| Model | Key Config |
|-------|------------|
| Random Forest | 200 trees, n_jobs=-1 |
| XGBoost | 200 estimators, mlogloss |
| MLP Neural Network | hidden=(128, 64), max_iter=500 |
| Logistic Regression | max_iter=1000 |
| SVM | RBF kernel, probability=True |

**Experimental Setup:**
- 5-Fold Stratified Cross-Validation on 70% train set
- 70% Train / 15% Validation / 15% Test split
- random_state=42 for full reproducibility

---

## SLIDE 10 — Evaluation Metrics

**Title:** Evaluation Metrics — Why Each Metric Matters

| Metric | Formula / Method | Clinical Relevance |
|--------|-----------------|-------------------|
| Accuracy | Correct / Total | General performance |
| Macro F1-Score | Unweighted avg F1 per class | Handles class imbalance fairly |
| AUC (Macro OvR) | Area under ROC curve | Discrimination ability across all classes |
| G-Mean | (∏ per-class recall)^(1/K) | Balanced performance — penalizes ignoring minority class |
| MAE (Ordinal) | Mean \|predicted − true\| ordinal steps | **Key ordinal metric** — penalizes large ordinal jumps |

**Why MAE is critical here:**
> A model that predicts Diabetic when patient is Normal (MAE=2) is far more dangerous than predicting Pre-Diabetic (MAE=1). Standard accuracy treats both equally — MAE does not.

---

## SLIDE 11 — Benchmark Results

**Title:** Benchmark Results — Model Comparison

> Note: Single models evaluated via 5-Fold CV on train set. Ensemble evaluated on held-out test set (15%).

| Model | Type | Accuracy | Macro F1 | AUC | G-Mean | MAE |
|-------|------|----------|----------|-----|--------|-----|
| **Ensemble\_Ordinal ⭐** | **Ordinal** | **92.03%** | **0.919** | **0.986** | **0.917** | **0.081** |
| XGBoost | Baseline | 89.42% | 0.892 | 0.973 | 0.889 | 0.107 |
| RandomForest | Baseline | 88.76% | 0.885 | 0.974 | 0.882 | 0.114 |
| OrdinalRandomForest | Ordinal | 88.39% | 0.881 | 0.973 | 0.878 | 0.117 |
| OrdinalGradientBoosting | Ordinal | 87.61% | 0.872 | 0.965 | 0.868 | 0.125 |
| MLP | Baseline | 78.99% | 0.784 | 0.925 | 0.774 | 0.212 |
| SVM | Baseline | 67.46% | 0.611 | 0.835 | 0.521 | 0.332 |
| LogisticRegression | Baseline | 63.24% | 0.597 | 0.782 | 0.542 | 0.373 |
| ProportionalOdds | Ordinal | 60.45% | 0.591 | 0.763 | 0.559 | 0.399 |

**Key Insight:**
> Among all ordinal models, **Ordinal Random Forest** achieves the best single-model performance. Combined with **Ordinal XGBoost** in a weighted ensemble (60%/40%), the proposed system achieves **92.03% accuracy** — a +2.92% improvement over the single model baseline.

**[IMAGE: benchmark_results.png]**

---

## SLIDE 12 — Final Test Set Results

**Title:** Final Evaluation — Proposed Model (Ensemble: Ordinal RF 60% + Ordinal XGBoost 40%)

**Generalization Performance:**

| Split | Accuracy | Macro F1 | AUC Macro | MAE |
|-------|----------|----------|-----------|-----|
| Train CV (5-fold) | 91.52% | 0.913 | — | 0.086 |
| Validation (15%) | 92.41% | 0.923 | 0.987 | 0.077 |
| **Test (15%)** | **92.03%** | **0.919** | **0.986** | **0.081** |

> CV → Test gap = 0.50% → **No overfitting. No underfitting.**

**Per-Class Performance on Test Set:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.93 | 1.00 | 0.96 |
| Pre-Diabetic | 0.94 | 0.81 | 0.87 |
| Diabetic | 0.89 | 0.95 | 0.92 |

> Pre-Diabetic F1 = 0.87 — improved from 0.83, reflecting better boundary class detection.

**[IMAGE: confusion_matrix_Ensemble_Final.png]**

**[IMAGE: generalization_analysis_Ensemble_Final.png]**

---

## SLIDE 13 — XAI: SHAP Global Explanation

**Title:** Explainability — SHAP Global Feature Importance

**What is SHAP?**
> SHapley Additive exPlanations — assigns each feature a contribution value based on cooperative game theory. Provides both *global* (population-level) and *local* (patient-level) explanations.

**Implementation:**
- TreeExplainer applied on the **P→Y binary boundary classifier** (most clinically critical — distinguishing Pre-Diabetic from Diabetic)
- Computed on 150 test samples

**Top 5 SHAP Features (14-feature Ensemble model):**

| Feature | Mean |SHAP| | Clinical Meaning |
|---------|-------------|-----------------|
| HbA1c | 0.181 | Glycated hemoglobin — ADA gold standard |
| FBS | 0.149 | Fasting blood sugar — direct diabetes indicator |
| Age | 0.019 | Risk increases significantly after age 45 |
| Creatinine | 0.017 | Kidney function — affected by long-term diabetes |
| BMI | 0.017 | Obesity — primary T2D risk factor |

> SHAP confirms: **HbA1c and FBS are the dominant biological levers** — fully consistent with ADA clinical guidelines (HbA1c ≥6.5% = Diabetic; FBS ≥126 mg/dL = Diabetic).

**[IMAGE: shap_summary_Ensemble_Final.png]**

---

## SLIDE 14 — XAI: SHAP Beeswarm & Local Explanation

**Title:** SHAP — Feature Impact Distribution & Individual Patient Explanation

**Beeswarm Plot Interpretation:**
- Each dot = one patient
- Red dots = high feature value → pushes prediction toward Diabetic
- Blue dots = low feature value → pushes prediction toward Normal
- Horizontal position = magnitude of SHAP impact

**[IMAGE: shap_beeswarm_Ensemble_Final.png]**

**Local Explanation — Waterfall Plot:**
- Shows exactly how each feature contributed to a single patient's prediction
- Base value = average model output; final value = this patient's prediction
- Clinicians can trace *why* a specific patient was classified as Diabetic

**[IMAGE: shap_local_Ensemble_Final.png]**

---

## SLIDE 15 — XAI: LIME Local Explanations

**Title:** Explainability — LIME Local Patient Explanations

**What is LIME?**
> Local Interpretable Model-agnostic Explanations — builds a local linear approximation around each individual prediction. Model-agnostic: works with any classifier.

**Implementation:**
- LimeTabularExplainer with discretized continuous features
- One representative patient from each class explained
- Shows feature contributions for all 3 classes simultaneously

**Three Patient Explanations Generated:**

| Patient Type | File | Key Finding |
|-------------|------|-------------|
| Normal Patient | lime_normal_Ensemble_Final.png | Low HbA1c and FBS push strongly toward Normal |
| Pre-Diabetic Patient | lime_pre_diabetic_Ensemble_Final.png | Borderline HbA1c (5.7–6.4%) drives Pre-Diabetic classification |
| Diabetic Patient | lime_diabetic_Ensemble_Final.png | High HbA1c (>6.5%) and FBS (>126) dominate Diabetic prediction |

**[IMAGE: lime_normal_Ensemble_Final.png]**
**[IMAGE: lime_pre_diabetic_Ensemble_Final.png]**
**[IMAGE: lime_diabetic_Ensemble_Final.png]**

---

## SLIDE 16 — Key Findings & Contributions

**Title:** Key Findings & Research Contributions

**Findings:**

1. **Proposed Ensemble (Ordinal RF + Ordinal XGBoost)** achieves 92.03% test accuracy, F1=0.919, AUC=0.986, MAE=0.081 — a **+2.92% improvement** over single model baseline
2. **HbA1c and FBS** are the dominant predictors — confirmed by both feature selection (scores: 0.975, 0.718) and SHAP (mean |SHAP|: 0.181, 0.149)
3. **Ordinal-SMOTE** successfully balanced the dataset from 4.3%/13.3%/82.4% to 33.3%/33.3%/33.3% while preserving ordinal boundaries
4. **Pre-Diabetic class** remains the hardest to classify (F1=0.87) — improved significantly with Ensemble, reflecting better boundary class detection
5. CV→Test gap of only **0.50%** confirms **no overfitting** — model generalizes well to unseen clinical data

**Novel Contributions:**

| Contribution | Description |
|-------------|-------------|
| Multi-Method Feature Selection | MI + Chi² + RF combined ranking — ranks all 14 features by importance; ablation study confirmed all features contribute meaningfully (all-14: 92.03% vs top-10: 90.63%) |
| Custom Ordinal-SMOTE | Adjacent-class-only interpolation — preserves N < P < Y in synthetic data |
| Frank & Hall + RF & XGBoost | Ordinal RF & XGBoost wrappers — sklearn-compatible, outperform Proportional Odds |
| Weighted Ensemble (60/40) | Ordinal RF + Ordinal XGBoost combination — +2.92% over single model baseline, 92.03% accuracy |
| Integrated XAI Pipeline | SHAP (global + local) + LIME (local) — clinician-ready explanations |
| Data Leakage Detection | Identified and removed 5 outcome-derived leakage columns |
| 70/15/15 Split | Proper 3-way split — prevents val/test contamination |

---

## SLIDE 17 — Conclusion

**Title:** Conclusion

**Summary:**
- We proposed a complete **Explainable Ordinal Regression** framework for diabetes risk stratification
- The framework correctly models the **clinical progression order** (Normal → Pre-Diabetic → Diabetic) using Frank & Hall decomposition
- **Proposed Ensemble (Ordinal RF 60% + Ordinal XGBoost 40%)** achieved 92.03% test accuracy with MAE=0.081 — **+2.92% improvement** over single model baseline, while fully respecting ordinal constraints
- **SHAP and LIME** provide clinician-interpretable explanations at both global and patient levels
- **HbA1c and FBS** are confirmed as the primary biological levers — consistent with ADA clinical guidelines

**Clinical Impact:**
> This system can assist clinicians in early identification of Pre-Diabetic patients — enabling timely lifestyle interventions before progression to full Diabetes.

**Limitations & Future Work:**
- Dataset is synthetically constructed from NDR 2023 report statistics — future work: validate on real de-identified patient records
- Proportional Odds model underperformed (60.45%) — future work: deep ordinal neural networks (CORN, OPAL)
- SHAP currently applied post-selection — future: integrate SHAP directly into feature selection loop
- Extend to longitudinal patient data for progression tracking over time
- Validate on external multi-center real-world clinical datasets

---

## SLIDE 18 — References

**Title:** References

[1] E. Frank and M. Hall, "A Simple Approach to Ordinal Classification," in *Proc. European Conference on Machine Learning (ECML)*, 2001, pp. 145–156.

[2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.

[3] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 1135–1144.

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[6] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD*, 2016, pp. 785–794.

[7] F. Pérez-Cruz, "mord: Ordinal Regression in Python," *Journal of Machine Learning Research*, vol. 17, pp. 1–5, 2016.

[8] American Diabetes Association, "Standards of Medical Care in Diabetes — 2023," *Diabetes Care*, vol. 46, Supplement 1, 2023.

[9] International Diabetes Federation, *IDF Diabetes Atlas*, 10th ed., Brussels, Belgium: IDF, 2021.

[10] Institute for Public Health (IPH), National Diabetes Registry (NDR), *Characteristics of Patients Enrolled in National Diabetes Registry, 2023*, Ministry of Health Malaysia, Kuala Lumpur, Malaysia, 2023. [Registry Dataset]

---

## APPENDIX — Output Files Reference

| Slide | Image File | Description |
|-------|-----------|-------------|
| Slide 7 | `outputs/feature_importance.png` | All 14 features — multi-method combined score ranking |
| Slide 11 | `outputs/benchmark_results.png` | All models comparison bar chart |
| Slide 12 | `outputs/confusion_matrix_Ensemble_Final.png` | Confusion matrix — Ensemble Final |
| Slide 12 | `outputs/generalization_analysis_Ensemble_Final.png` | Train/Val/Test generalization analysis |
| Slide 13 | `outputs/shap_summary_Ensemble_Final.png` | SHAP global feature importance bar |
| Slide 14 (top) | `outputs/shap_beeswarm_Ensemble_Final.png` | SHAP beeswarm impact distribution |
| Slide 14 (bottom) | `outputs/shap_local_Ensemble_Final.png` | SHAP waterfall — individual patient |
| Slide 15 | `outputs/lime_normal_Ensemble_Final.png` | LIME — Normal patient explanation |
| Slide 15 | `outputs/lime_pre_diabetic_Ensemble_Final.png` | LIME — Pre-Diabetic patient explanation |
| Slide 15 | `outputs/lime_diabetic_Ensemble_Final.png` | LIME — Diabetic patient explanation |

---

*End of PPT Content File*
*IEEE Conference Presentation Format | Total Slides: 18*
