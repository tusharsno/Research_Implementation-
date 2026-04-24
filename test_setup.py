#!/usr/bin/env python3
"""
test_setup.py
Quick test to verify dependencies and basic preprocessing.
"""

import sys

print("Testing dependencies...")
print("-" * 50)

# Test imports
try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except ImportError as e:
    print("✗ numpy:", e)
    sys.exit(1)

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except ImportError as e:
    print("✗ pandas:", e)
    sys.exit(1)

try:
    import sklearn
    print("✓ scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("✗ scikit-learn:", e)
    sys.exit(1)

try:
    import imblearn
    print("✓ imbalanced-learn:", imblearn.__version__)
except ImportError as e:
    print("✗ imbalanced-learn:", e)
    sys.exit(1)

try:
    import mord
    print("✓ mord: installed")
except ImportError as e:
    print("✗ mord:", e)
    sys.exit(1)

try:
    import xgboost as xgb
    print("✓ xgboost:", xgb.__version__)
except ImportError as e:
    print("✗ xgboost:", e)
    sys.exit(1)

try:
    import matplotlib
    print("✓ matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("✗ matplotlib:", e)
    sys.exit(1)

try:
    import seaborn as sns
    print("✓ seaborn:", sns.__version__)
except ImportError as e:
    print("✗ seaborn:", e)
    sys.exit(1)

try:
    import shap
    print("✓ shap:", shap.__version__)
except ImportError as e:
    print("⚠ shap:", e, "(optional for XAI)")

try:
    import lime
    print("✓ lime: installed")
except ImportError as e:
    print("⚠ lime:", e, "(optional for XAI)")

print("-" * 50)
print("\n✓ All core dependencies installed!\n")

# Test basic preprocessing
print("Testing preprocessing on actual dataset...")
print("-" * 50)

try:
    from data_preprocessing import load_data, drop_admin_columns, encode_labels, encode_categoricals
    
    df = load_data("diabetes_balanced_dataset (1).csv")
    print(f"✓ Loaded dataset: {df.shape}")
    
    df = drop_admin_columns(df)
    print(f"✓ Dropped admin columns")
    
    df = encode_labels(df, "Risk_Level")
    print(f"✓ Encoded labels")
    
    df = encode_categoricals(df)
    print(f"✓ Encoded categoricals: {df.shape}")
    
    print("\n✓ Preprocessing test passed!")
    print(f"  Final features: {[c for c in df.columns if c != 'Risk_Level'][:10]}...")
    
except Exception as e:
    print(f"\n✗ Preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Setup verification complete! Ready to run main.py")
print("=" * 50)
