import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os
import psutil
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "features.csv"
TRAIN_RATIO = 0.8  # First 80% = train, last 20% = test (TIME-BASED, NO SHUFFLING)

# XGBoost hyperparameters
n_estimators = 396
max_depth = 2
learning_rate = 0.0035827
subsample = 0.6848
min_child_weight = 15
colsample_bytree = 0.5757
gamma = 0.2493
reg_alpha = 1.6145
reg_lambda = 1.6985

# ================================
# HARDWARE DETECTION
# ================================

cpu_count = psutil.cpu_count(logical=False)
N_THREADS = max(1, cpu_count - 1) if cpu_count else 1
print(f"Detected {cpu_count} physical CPU cores — using {N_THREADS} threads")

DEVICE = 'cpu'
TREE_METHOD = 'hist'

try:
    test_matrix = xgb.DMatrix(np.random.rand(100, 5), label=np.random.rand(100))
    xgb.train(
        {'tree_method': 'gpu_hist', 'device': 'cuda', 'objective': 'reg:squarederror'},
        test_matrix,
        num_boost_round=1,
        verbose_eval=False
    )
    DEVICE = 'cuda'
    TREE_METHOD = 'gpu_hist'
    print("✓ GPU detected - using CUDA acceleration")
except Exception:
    print("GPU not available - using CPU (tree_method=hist)")

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
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buy_label', 'sell_label', 'idk_label', 'Timestamp', 'DateTime']
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
# Keep strong imbalance handling per label, without diluting to a shared normalized value.
buy_weight_raw = 1 / buy_ratio_train if buy_ratio_train > 0 else 1
sell_weight_raw = 1 / sell_ratio_train if sell_ratio_train > 0 else 1

print(f"\nClass weights (raw inverse frequency):")
print(f"BUY positive weight approx: {buy_weight_raw:.2f}")
print(f"SELL positive weight approx: {sell_weight_raw:.2f}")

# Optional saturation factor to upweight rare positives even more
WEIGHT_FACTOR = 1.0  # can tune (1.0, 2.0, 5.0)
buy_weight = buy_weight_raw * WEIGHT_FACTOR
sell_weight = sell_weight_raw * WEIGHT_FACTOR

print(f"\nEffective class weight factors (for sample_weight in training):")
print(f"BUY class positive weight: {buy_weight:.2f}")
print(f"SELL class positive weight: {sell_weight:.2f}")

# ================================
# TRAIN XGBOOST (MULTI-OUTPUT) - GPU ACCELERATED
# ================================

print(f"\n===== TRAINING XGBOOST ON {DEVICE.upper()} =====")

# One model per output (BUY and SELL)
models = {}

for idx, label_name in enumerate(tqdm(['buy', 'sell'], desc="Training models")):
    
    y_train_label = y_train[:, idx]
    
    # Class weight for this label
    class_weight = buy_weight if label_name == 'buy' else sell_weight
    sample_weight = np.where(y_train_label == 1, class_weight, 1.0)
    
    # Create DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weight, feature_names=feature_cols)
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'min_child_weight': min_child_weight,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'tree_method': TREE_METHOD,
        'device': DEVICE,
        'nthread': N_THREADS,
    }
    
    # Custom progress callback
    pbar = tqdm(total=n_estimators, desc=f"  {label_name.upper()} trees", leave=False)

    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        callbacks=[TqdmCallback()],
        verbose_eval=False
    )

    pbar.close()
    
    models[label_name] = booster

# ================================
# SAVE MODELS
# ================================

print("\n===== SAVING MODELS =====")

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

for label_name, booster in models.items():
    model_path = os.path.join(models_dir, f"xgboost_{label_name}_model.joblib")
    joblib.dump(booster, model_path)
    print(f"✓ Saved: {model_path}")

# Also save feature column names for later use
feature_names_path = os.path.join(models_dir, "feature_names.joblib")
joblib.dump(feature_cols, feature_names_path)
print(f"✓ Saved: {feature_names_path}")

# ================================
# EVALUATION
# ================================

print("\n===== MODEL EVALUATION =====")

dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

for label_name in ['buy', 'sell']:
    print(f"\n{label_name.upper()} MODEL:")
    
    booster = models[label_name]
    y_test_label = y_test[:, 0 if label_name == 'buy' else 1]
    
    # Predictions
    y_proba = booster.predict(dtest)
    y_pred = (y_proba > 0.5).astype(int)
    
    # Metrics
    print(classification_report(y_test_label, y_pred))
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test_label, y_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except:
        print("ROC-AUC: N/A (not enough positive samples)")
    
    # Feature importance (top 10)
    importance = booster.get_score(importance_type='weight')
    feature_importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features:")
    print(feature_importance.head(10))

# ================================
# PREDICTIONS ON TEST SET (for later use)
# ================================

print("\n===== GENERATING PREDICTIONS =====")

buy_proba = models['buy'].predict(dtest)
sell_proba = models['sell'].predict(dtest)

predictions_df = test_df.copy()
predictions_df['buy_prob'] = buy_proba
predictions_df['sell_prob'] = sell_proba

# Decision logic with uncertainty filtering (as designed)
predictions_df['decision'] = 'NO_TRADE'
predictions_df.loc[(buy_proba > 0.5) & (sell_proba < 0.5), 'decision'] = 'BUY'
predictions_df.loc[(sell_proba > 0.5) & (buy_proba < 0.5), 'decision'] = 'SELL'

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
buy_importance_dict = models['buy'].get_score(importance_type='weight')
buy_importance = pd.DataFrame({'feature': list(buy_importance_dict.keys()), 'importance': list(buy_importance_dict.values())}).sort_values('importance', ascending=False).head(10)
ax3.barh(buy_importance['feature'], buy_importance['importance'])
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Features - BUY Model')

# Plot 4: Feature importance (SELL)
ax4 = axes[1, 1]
sell_importance_dict = models['sell'].get_score(importance_type='weight')
sell_importance = pd.DataFrame({'feature': list(sell_importance_dict.keys()), 'importance': list(sell_importance_dict.values())}).sort_values('importance', ascending=False).head(10)
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
    import xgboost as xgb
    
    buy_model = joblib.load("models/xgboost_buy_model.joblib")
    sell_model = joblib.load("models/xgboost_sell_model.joblib")
    feature_cols = joblib.load("models/feature_names.joblib")
    
    # Use on new data
    dmatrix = xgb.DMatrix(new_data[feature_cols].values)
    buy_proba = buy_model.predict(dmatrix)
    sell_proba = sell_model.predict(dmatrix)
""")