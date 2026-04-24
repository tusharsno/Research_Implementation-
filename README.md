# Diabetes Risk Prediction System
## Ordinal Regression + Explainable AI (XAI)

### Clinical Progression Model
**Normal (0) → Pre-Diabetic (1) → Diabetic (2)**

---

## 🎯 Project Overview

This implementation provides a complete pipeline for diabetes risk prediction using:
- **Ordinal Regression** (respects disease progression order)
- **Multi-method Feature Selection** (MI, Chi-Square, Random Forest)
- **Ordinal-SMOTE** (class balancing preserving order)
- **XAI Framework** (SHAP + LIME for clinical interpretability)

### Key Features
✅ Custom Ordinal Random Forest (Frank & Hall decomposition)  
✅ Ordinal XGBoost variant  
✅ Proportional Odds model (mord library)  
✅ Benchmark against 5 baseline classifiers  
✅ Global & local explanations (SHAP, LIME)  
✅ Highlights HbA1c and lipid metabolism as biological levers  

---

## 📦 Installation

### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda create -n diabetes_pred python=3.10
conda activate diabetes_pred
pip install -r requirements.txt
```

### Option 3: System-wide (if venv not available)
```bash
pip install --user -r requirements.txt
```

---

## 🚀 Usage

### Quick Start (with synthetic data)
```bash
python main.py
```

### With Your Dataset
```bash
python main.py --data path/to/your_data.csv --target Outcome --top_k 10
```

### Advanced Options
```bash
python main.py \
    --data diabetes_data.csv \
    --target Outcome \
    --top_k 12 \
    --cv_folds 5 \
    --best_model Ordinal_XGBoost \
    --output_dir results \
    --skip_xai  # Skip XAI analysis for faster run
```

---

## 📊 Expected Dataset Format

Your CSV should have:
- **Clinical features**: Age, BMI, Glucose, HbA1c, Cholesterol, Creatinine, etc.
- **Target column**: `Outcome` with values:
  - `"N"` or `0` → Normal
  - `"P"` or `1` → Pre-Diabetic
  - `"Y"` or `2` → Diabetic

Example:
```csv
Age,BMI,Glucose,HbA1c,Cholesterol,Creatinine,SystolicBP,DiastolicBP,Outcome
45,28.5,110,5.8,195,0.9,125,82,N
52,31.2,145,6.5,220,1.1,138,88,P
61,34.8,180,8.2,245,1.4,155,95,Y
```

---

## 📁 Project Structure

```
Research_Implementation/
├── main.py                    # Pipeline orchestrator
├── data_preprocessing.py      # Cleaning, normalization, Ordinal-SMOTE
├── feature_selection.py       # MI, Chi-Square, RF importance
├── ordinal_models.py          # Custom Ordinal RF, XGBoost, baselines
├── model_training.py          # Training, CV, benchmarking
├── explainability.py          # SHAP + LIME integration
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── outputs/                   # Generated plots and results
    ├── feature_importance.png
    ├── benchmark_results.png
    ├── confusion_matrix_*.png
    ├── shap_summary_*.png
    ├── shap_beeswarm_*.png
    ├── lime_*.png
    └── benchmark_summary.csv
```

---

## 🔬 Models Implemented

### Ordinal Models (Respect N < P < Y order)
1. **Proportional Odds** (mord.LogisticAT)
2. **Ordinal Random Forest** (Custom Frank & Hall)
3. **Ordinal XGBoost** (Custom binary decomposition)

### Baseline Models (Standard multiclass)
1. Random Forest
2. XGBoost
3. MLP Neural Network
4. Logistic Regression
5. SVM (RBF kernel)

---

## 📈 Evaluation Metrics

- **Weighted F1-Score** (handles class imbalance)
- **Macro AUC** (OvR multi-class)
- **Mean Absolute Error (MAE)** (ordinal-specific)
- **Per-class Precision/Recall/F1**
- **Confusion Matrix** (with percentages)

---

## 🧠 XAI Analysis

### SHAP (SHapley Additive exPlanations)
- **Global**: Feature importance across all predictions
- **Beeswarm**: Impact distribution per feature
- **Local**: Waterfall plot for individual patient

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local**: Feature contributions per class for specific patient
- **Multi-sample**: Representative explanations from each class

---

## 🎓 Research Paper Implementation Notes

### Why Ordinal Regression?
Standard classification treats classes as independent. But diabetes progression has a natural order:
```
Normal → Pre-Diabetic → Diabetic
```
Ordinal models penalize misclassifications proportionally (N→Y is worse than N→P).

### Ordinal-SMOTE Strategy
- Balances only between **adjacent classes** (N↔P, P↔Y)
- Preserves clinical progression order
- Specifically targets under-represented Pre-Diabetic class

### Key Biological Levers (from your PPTX)
1. **HbA1c** (glycated hemoglobin) — gold standard for diabetes diagnosis
2. **Lipid metabolism** (Cholesterol, Triglycerides)
3. **Glucose** (fasting blood sugar)
4. **BMI** (obesity indicator)

---

## 🔧 Troubleshooting

### Import Errors
```bash
# If mord fails to install
pip install --upgrade pip setuptools wheel
pip install mord

# If SHAP fails
pip install shap --no-build-isolation
```

### Memory Issues (Large Dataset)
```python
# In explainability.py, reduce max_samples
shap_analyzer.compute_shap_values(X_test, max_samples=50)
```

### Slow XAI Analysis
```bash
# Skip XAI for quick benchmarking
python main.py --skip_xai
```

---

## 📚 References

- **Frank & Hall (2001)**: "A Simple Approach to Ordinal Classification"
- **mord library**: https://github.com/fabianp/mord
- **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- **LIME**: Ribeiro et al. (2016) - "Why Should I Trust You?"

---

## 🤝 Contributing

This is a research implementation. To extend:
1. Add new ordinal models in `ordinal_models.py`
2. Implement custom feature engineering in `data_preprocessing.py`
3. Add new XAI methods in `explainability.py`

---

## 📄 License

MIT License - Free for research and educational purposes.

---

## 👨‍💻 Author

Developed for healthcare analytics research focusing on explainable ordinal regression.

**Contact**: Replace with your details

---

## 🎯 Expected Results (Synthetic Data)

Based on your PPTX findings:
- **Ordinal XGBoost**: F1 ≈ 0.95-1.00
- **Ordinal Random Forest**: F1 ≈ 0.92-0.98
- **Proportional Odds**: F1 ≈ 0.85-0.92
- **Baseline Models**: F1 ≈ 0.80-0.95

Top features: HbA1c, Glucose, BMI, Cholesterol, Age

---

**Happy Modeling! 🚀**
