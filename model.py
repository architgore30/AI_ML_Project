import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "features.csv"
TRAIN_RATIO = 0.8  # First 80% = train, last 20% = test (TIME-BASED, NO SHUFFLING)

# XGBoost hyperparameters
n_estimators = 100
max_depth = 5
learning_rate = 0.1
subsample = 0.8

# ================================
# LOAD DATA
# ================================

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} samples")

# Drop rows with NaN features (warmup period)
df_clean = df.dropna()
print(f"After removing NaN warmup period: {len(df_clean)} samples")

# ================================
# TIME-BASED SPLIT (NO SHUFFLING - CRITICAL)
# ================================

split_idx = int(len(df_clean) * TRAIN_RATIO)

train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

print(f"\nTrain: {len(train_df)} samples")
print(f"Test: {len(test_df)} samples")

# ================================
# SEPARATE FEATURES AND LABELS
# ================================

# List all feature columns (exclude OHLCV, volume, and labels)
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buy_label', 'sell_label', 'idk_label', 'Timestamp']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

print(f"\nUsing {len(feature_cols)} features:")
print(feature_cols[:10], "... (showing first 10)")

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# For multi-output: we have buy_label and sell_label (idk_label is derived)
y_train = train_df[['buy_label', 'sell_label']].values
y_test = test_df[['buy_label', 'sell_label']].values

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ================================
# LABEL DISTRIBUTION & CLASS WEIGHTS
# ================================

print("\n===== TRAINING SET LABEL DISTRIBUTION =====")
buy_ratio_train = y_train[:, 0].sum() / len(y_train)
sell_ratio_train = y_train[:, 1].sum() / len(y_train)
print(f"BUY ratio: {buy_ratio_train:.4f}")
print(f"SELL ratio: {sell_ratio_train:.4f}")

# Compute class weights (to handle imbalance)
buy_weight = 1 / buy_ratio_train if buy_ratio_train > 0 else 1
sell_weight = 1 / sell_ratio_train if sell_ratio_train > 0 else 1
buy_weight /= (buy_weight + sell_weight)
sell_weight /= (buy_weight + sell_weight)

print(f"\nClass weights (normalized):")
print(f"BUY: {buy_weight:.4f}")
print(f"SELL: {sell_weight:.4f}")

# ================================
# TRAIN XGBOOST (MULTI-OUTPUT)
# ================================

print("\n===== TRAINING XGBOOST =====")

# One model per output (BUY and SELL)
models = {}

for idx, label_name in enumerate(tqdm(['buy', 'sell'], desc="Training models")):
    # print(f"\nTraining {label_name.upper()} model...")
    
    y_train_label = y_train[:, idx]
    
    # Class weight for this label
    class_weight = buy_weight if label_name == 'buy' else sell_weight
    sample_weight = np.where(y_train_label == 1, class_weight, 1 - class_weight)
    
    model = GradientBoostingClassifier(
        n_estimators=1,  # Start with 1 tree
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        warm_start=True,  # Allow incremental training
        verbose=0
    )
    
    # Train incrementally, showing progress
    for tree_idx in tqdm(range(n_estimators), desc=f"  {label_name.upper()} trees", leave=False):
        model.n_estimators = tree_idx + 1
        model.fit(X_train, y_train_label, sample_weight=sample_weight)
    
    models[label_name] = model
    
    # print(f"✓ {label_name.upper()} model trained")

# ================================
# SAVE MODELS
# ================================

print("\n===== SAVING MODELS =====")

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

for label_name, model in models.items():
    model_path = os.path.join(models_dir, f"xgboost_{label_name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"✓ Saved: {model_path}")

# Also save feature column names for later use
feature_names_path = os.path.join(models_dir, "feature_names.joblib")
joblib.dump(feature_cols, feature_names_path)
print(f"✓ Saved: {feature_names_path}")

# ================================
# EVALUATION
# ================================

print("\n===== MODEL EVALUATION =====")

for label_name in ['buy', 'sell']:
    print(f"\n{label_name.upper()} MODEL:")
    
    model = models[label_name]
    y_test_label = y_test[:, 0 if label_name == 'buy' else 1]
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(classification_report(y_test_label, y_pred))
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test_label, y_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except:
        print("ROC-AUC: N/A (not enough positive samples)")
    
    # Feature importance (top 10)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features:")
    print(feature_importance.head(10))

# ================================
# PREDICTIONS ON TEST SET (for later use)
# ================================

print("\n===== GENERATING PREDICTIONS =====")

buy_proba = models['buy'].predict_proba(X_test)[:, 1]
sell_proba = models['sell'].predict_proba(X_test)[:, 1]

predictions_df = test_df.copy()
predictions_df['buy_prob'] = buy_proba
predictions_df['sell_prob'] = sell_proba

# Decision logic with uncertainty filtering (as designed)
predictions_df['decision'] = 'NO_TRADE'
predictions_df.loc[(buy_proba > 0.7) & (sell_proba < 0.3), 'decision'] = 'BUY'
predictions_df.loc[(sell_proba > 0.7) & (buy_proba < 0.3), 'decision'] = 'SELL'

print(f"\nDecision distribution:")
print(predictions_df['decision'].value_counts())

# Save predictions
predictions_df.to_csv('predictions_test.csv', index=False)
print("\n✓ Predictions saved to: predictions_test.csv")

# ================================
# VISUALIZATION
# ================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: BUY probability distribution
ax1 = axes[0, 0]
ax1.hist(buy_proba[y_test[:, 0] == 1], bins=30, alpha=0.6, label='Actual BUY', color='green')
ax1.hist(buy_proba[y_test[:, 0] == 0], bins=30, alpha=0.6, label='No BUY', color='red')
ax1.set_xlabel('BUY Probability')
ax1.set_ylabel('Frequency')
ax1.set_title('BUY Model - Prediction Distribution')
ax1.legend()

# Plot 2: SELL probability distribution
ax2 = axes[0, 1]
ax2.hist(sell_proba[y_test[:, 1] == 1], bins=30, alpha=0.6, label='Actual SELL', color='red')
ax2.hist(sell_proba[y_test[:, 1] == 0], bins=30, alpha=0.6, label='No SELL', color='green')
ax2.set_xlabel('SELL Probability')
ax2.set_ylabel('Frequency')
ax2.set_title('SELL Model - Prediction Distribution')
ax2.legend()

# Plot 3: Feature importance (BUY)
ax3 = axes[1, 0]
buy_importance = pd.DataFrame({'feature': feature_cols, 'importance': models['buy'].feature_importances_}).sort_values('importance', ascending=False).head(10)
ax3.barh(buy_importance['feature'], buy_importance['importance'])
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Features - BUY Model')

# Plot 4: Feature importance (SELL)
ax4 = axes[1, 1]
sell_importance = pd.DataFrame({'feature': feature_cols, 'importance': models['sell'].feature_importances_}).sort_values('importance', ascending=False).head(10)
ax4.barh(sell_importance['feature'], sell_importance['importance'])
ax4.set_xlabel('Importance')
ax4.set_title('Top 10 Features - SELL Model')

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: xgboost_results.png")
plt.show()

# ================================
# UNCERTAINTY FILTER ANALYSIS
# ================================

print("\n===== UNCERTAINTY FILTER IMPACT =====")

# Before filtering
trades_before = len(predictions_df[predictions_df['decision'] != 'NO_TRADE'])
print(f"Trades before filtering: {trades_before} ({trades_before/len(predictions_df)*100:.2f}%)")

# Count by type
print(f"BUY signals: {(predictions_df['decision'] == 'BUY').sum()}")
print(f"SELL signals: {(predictions_df['decision'] == 'SELL').sum()}")
print(f"NO_TRADE: {(predictions_df['decision'] == 'NO_TRADE').sum()}")

print("\n✓ Training complete!")

# ================================
# HOW TO USE SAVED MODELS
# ================================

print("\n===== LOADING MODELS FOR BACKTESTING =====")
print("""
To load these models in a backtesting script:

    import joblib
    
    buy_model = joblib.load("models/xgboost_buy_model.joblib")
    sell_model = joblib.load("models/xgboost_sell_model.joblib")
    feature_cols = joblib.load("models/feature_names.joblib")
    
    # Use on new data
    X_new = new_data[feature_cols].values
    buy_proba = buy_model.predict_proba(X_new)[:, 1]
    sell_proba = sell_model.predict_proba(X_new)[:, 1]
""")
