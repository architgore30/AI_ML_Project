"""
Predictions Generator for XGBoost Models
Purpose: Generate predictions_test.csv for a specific model tier (XGBoost GPU-trained)
Output saved to the specified models folder
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sys

# ================================
# CONFIGURATION
# ================================

# Specify which model folder to use (models, models-2, models-3, or models-4)
MODELS_PATH = "models-4"  # Change this to models-2 or models-3 for other tiers

DATA_PATH = "features.csv"
TRAIN_RATIO = 0.8  # First 80% = train, last 20% = test (TIME-BASED, NO SHUFFLING)

# Decision thresholds (0.5 = equal confidence in both signals to avoid trading)
THRESHOLD = 0.5

# ================================
# LOAD DATA
# ================================

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} samples")

# Drop rows with NaN features (warmup period)
df_clean = df.dropna()
print(f"After removing NaN warmup period: {len(df_clean)} samples")

# ================================
# TIME-BASED SPLIT (NO SHUFFLING - CRITICAL)
# ================================

split_idx = int(len(df_clean) * TRAIN_RATIO)
test_df = df_clean.iloc[split_idx:]

print(f"Test set: {len(test_df)} samples")

# ================================
# PREPARE FEATURES
# ================================

exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buy_label', 'sell_label', 'idk_label', 'Timestamp', 'DateTime']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X_test = test_df[feature_cols].values

print(f"Using {len(feature_cols)} features for prediction")

# ================================
# LOAD XGBOOST MODELS
# ================================

print(f"\nLoading XGBoost models from {MODELS_PATH}...")

try:
    buy_model = joblib.load(f"{MODELS_PATH}/xgboost_buy_model.joblib")
    sell_model = joblib.load(f"{MODELS_PATH}/xgboost_sell_model.joblib")
    print(f"✓ Models loaded successfully")
    print(f"  BUY model type: {type(buy_model).__name__}")
    print(f"  SELL model type: {type(sell_model).__name__}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

# ================================
# GENERATE PREDICTIONS (XGBoost)
# ================================

print("\n===== GENERATING PREDICTIONS =====")

try:
    # Convert to DMatrix for XGBoost (GPU acceleration if available)
    # Must provide feature_names so XGBoost knows what data to expect
    dmatrix_test = xgb.DMatrix(X_test, feature_names=feature_cols)
    
    # Get probabilities from XGBoost Booster
    # predict() returns probabilities directly if trained with binary:logistic
    buy_proba = buy_model.predict(dmatrix_test)
    sell_proba = sell_model.predict(dmatrix_test)
    
    print(f"✓ Predictions generated")
    print(f"  BUY proba range: {buy_proba.min():.4f} - {buy_proba.max():.4f}")
    print(f"  SELL proba range: {sell_proba.min():.4f} - {sell_proba.max():.4f}")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    sys.exit(1)

# ================================
# CREATE PREDICTIONS DATAFRAME
# ================================

predictions_df = test_df.copy()
predictions_df['buy_prob'] = buy_proba
predictions_df['sell_prob'] = sell_proba

# Decision logic with uncertainty filtering
# Threshold=0.5 means: only trade if ONE signal > 0.5 AND the other < 0.5
predictions_df['decision'] = 'NO_TRADE'
predictions_df.loc[(buy_proba > THRESHOLD) & (sell_proba < THRESHOLD), 'decision'] = 'BUY'
predictions_df.loc[(sell_proba > THRESHOLD) & (buy_proba < THRESHOLD), 'decision'] = 'SELL'

print(f"\n===== DECISION DISTRIBUTION =====")
print(predictions_df['decision'].value_counts())
print(f"\nDecision threshold: >{THRESHOLD:.2f} for buy/sell, <{THRESHOLD:.2f} for the other signal")

# ================================
# SAVE PREDICTIONS
# ================================

output_path = f"{MODELS_PATH}/predictions_test.csv"
predictions_df.to_csv(output_path, index=False)
print(f"\n✓ Predictions saved to: {output_path}")
print(f"  Columns: {', '.join(predictions_df.columns.tolist()[:5])}... ({len(predictions_df.columns)} total)")
print(f"  Rows: {len(predictions_df)}")