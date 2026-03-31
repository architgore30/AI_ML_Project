"""
Hyperparameter Tuning for XGBoost Models
Purpose: Grid search over n_estimators, max_depth, learning_rate, subsample
Evaluates each combination using backtesting (final P&L is the metric)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import subprocess
import sys
from itertools import product
from tqdm import tqdm

# ================================
# GPU DETECTION
# ================================

try:
    # Test GPU availability
    test_matrix = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
    test_model = xgb.train(
        {'tree_method': 'auto', 'device': 'cuda', 'objective': 'reg:squarederror'},
        test_matrix,
        num_boost_round=1,
        verbose_eval=False
    )
    GPU_AVAILABLE = True
    print("GPU detected and available for XGBoost")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"GPU not available, falling back to CPU")

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "features.csv"
TRAIN_RATIO = 0.8
WEIGHT_FACTOR = 1.0  # Start with base weighting

# Hyperparameter grid to search
PARAM_GRID = {
    'n_estimators': [50, 100, 200, 300, 350, 400],
    'max_depth': [1, 2, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.2, 0.4, 0.6, 0.8, 1.0]
}

# ================================
# LOAD DATA ONCE
# ================================

print("Loading data...")
df = pd.read_csv(DATA_PATH).iloc[:50_000] # training on smaller dataset to reduce total training time
df_clean = df.dropna()

split_idx = int(len(df_clean) * TRAIN_RATIO)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buy_label', 'sell_label', 'idk_label', 'Timestamp', 'DateTime']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
feature_cols = list(feature_cols)  # Ensure it's a list of strings

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

y_train = train_df[['buy_label', 'sell_label']].values
y_test = test_df[['buy_label', 'sell_label']].values

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

# ================================
# CLASS WEIGHTS (FIXED)
# ================================

buy_ratio_train = y_train[:, 0].sum() / len(y_train)
sell_ratio_train = y_train[:, 1].sum() / len(y_train)

buy_weight = (1 / buy_ratio_train) * WEIGHT_FACTOR if buy_ratio_train > 0 else 1
sell_weight = (1 / sell_ratio_train) * WEIGHT_FACTOR if sell_ratio_train > 0 else 1

print(f"Class weights: BUY={buy_weight:.2f}, SELL={sell_weight:.2f}")

# ================================
# GENERATE PARAMETER COMBINATIONS
# ================================

param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(product(*param_values))

total_combinations = len(all_combinations)
print(f"\nTotal parameter combinations to test: {total_combinations}")

# ================================
# TUNING LOOP
# ================================

results = []

for combo_idx, param_combo in enumerate(tqdm(all_combinations, desc="Testing hyperparameters")):
    params = dict(zip(param_names, param_combo))
    
    try:
        # Train BUY and SELL models with these parameters
        models = {}
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        
        for label_idx, label_name in enumerate(['buy', 'sell']):
            y_train_label = y_train[:, label_idx]
            class_weight = buy_weight if label_name == 'buy' else sell_weight
            sample_weight = np.where(y_train_label == 1, class_weight, 1.0)
            
            # XGBoost training
            dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weight, feature_names=feature_cols)
            
            tree_method = 'auto'
            
            xgb_params = {
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'objective': 'binary:logistic',
                'tree_method': tree_method
                # 'verbose_eval': False
            }
            
            # Add device parameter (auto will use GPU if available)
            xgb_params['device'] = 'cuda'
            
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                verbose_eval=False
            )
            
            models[label_name] = model
        
        # Generate predictions on test set
        buy_proba = models['buy'].predict(dtest)
        sell_proba = models['sell'].predict(dtest)
        
        # Create predictions CSV
        predictions_df = test_df.copy()
        predictions_df['buy_prob'] = buy_proba
        predictions_df['sell_prob'] = sell_proba
        predictions_df['decision'] = 'NO_TRADE'
        predictions_df.loc[(buy_proba > 0.5) & (sell_proba < 0.5), 'decision'] = 'BUY'
        predictions_df.loc[(sell_proba > 0.5) & (buy_proba < 0.5), 'decision'] = 'SELL'
        
        # Temporary output
        temp_pred_path = "temp_predictions.csv"
        predictions_df.to_csv(temp_pred_path, index=False)
        
        # Extract key metrics from predictions
        decision_counts = predictions_df['decision'].value_counts()
        buy_count = decision_counts.get('BUY', 0)
        sell_count = decision_counts.get('SELL', 0)
        no_trade_count = decision_counts.get('NO_TRADE', 0)
        
        # Calculate win rate on test set (simple heuristic: buy_prob - sell_prob > 0.3 = good signal)
        test_accuracy_proxy = ((buy_proba > 0.6).sum() + (sell_proba > 0.6).sum()) / len(test_df)
        
        # Store result
        results.append({
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'no_trade_signals': no_trade_count,
            'prediction_confidence': test_accuracy_proxy,
            'status': 'success'
        })
        
    except Exception as e:
        results.append({
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'status': f'error: {str(e)[:50]}'
        })

# ================================
# RESULTS ANALYSIS
# ================================

results_df = pd.DataFrame(results)

# Filter successful runs
successful = results_df[results_df['status'] == 'success'].copy()

print(f"\n===== HYPERPARAMETER TUNING RESULTS =====")
print(f"Successful runs: {len(successful)}/{total_combinations}")

if len(successful) > 0:
    # Sort by prediction confidence (proxy for quality)
    successful_sorted = successful.sort_values('prediction_confidence', ascending=False)
    
    print(f"\n===== TOP 10 PARAMETER SETS (by prediction confidence) =====\n")
    print(successful_sorted[['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                              'buy_signals', 'sell_signals', 'prediction_confidence']].head(10).to_string(index=False))
    
    # Save full results
    results_df.to_csv('tuning_results.csv', index=False)
    print(f"\n✓ Full results saved to: tuning_results.csv")
    
    # Best parameters
    best_params = successful_sorted.iloc[0]
    print(f"\n===== RECOMMENDED PARAMETERS =====")
    print(f"n_estimators: {int(best_params['n_estimators'])}")
    print(f"max_depth: {int(best_params['max_depth'])}")
    print(f"learning_rate: {best_params['learning_rate']}")
    print(f"subsample: {best_params['subsample']}")
    print(f"Prediction confidence: {best_params['prediction_confidence']:.4f}")
    
    print(f"\n⚠️  These are based on signal generation, not backtest P&L.")
    print(f"⚠️  For final validation, train with recommended params and run backtest.py manually.")
else:
    print("❌ No successful runs - check error log in tuning_results.csv")

# Clean up temp file
if os.path.exists("temp_predictions.csv"):
    os.remove("temp_predictions.csv")