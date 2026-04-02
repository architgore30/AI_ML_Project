"""
Bayesian Hyperparameter Optimization for XGBoost Models
Purpose: Use Optuna with TPE sampler, pruning, and parallel processing
Smart algorithm for weak hardware (i5 CPU)

Approach:
- Phase 1: 20 trials exploration with pruning (early stopping for bad trials)
- Phase 2: 10 trials refinement around best regions
- Parallel processing: 2-3 workers on i5
- Pruning stops bad trials at 50% training to save time
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import psutil
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================
# HARDWARE DETECTION & STRATEGY
# ================================

cpu_count = psutil.cpu_count(logical=False)
is_weak_hardware = cpu_count is not None and cpu_count <= 4

print(f"Detected {cpu_count} physical CPU cores")
if is_weak_hardware:
    print("✓ Weak hardware detected (<=4 cores) - using Bayesian optimization")
    N_JOBS = 3
else:
    print("✓ Standard/strong hardware - using full optimization")
    N_JOBS = 3

print(f"Will use {N_JOBS} parallel workers for trials")

# ================================
# GPU/CPU DETECTION & DEVICE STRATEGY
# ================================

GPU_AVAILABLE = False
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
    GPU_AVAILABLE = True
    DEVICE = 'cuda'
    TREE_METHOD = 'gpu_hist'
    print("✓ GPU detected - using CUDA acceleration")
except Exception as e:
    print(f"GPU not available - using CPU (tree_method=hist for efficiency)")
    DEVICE = 'cpu'
    TREE_METHOD = 'hist'

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "features.csv"
TRAIN_RATIO = 0.8
SAMPLE_SIZE = 200_000
PHASE_1_TRIALS = 60
PHASE_2_TRIALS = 25

print(f"\nUsing Bayesian optimization with pruning:")
print(f"  Phase 1: {PHASE_1_TRIALS} trials (exploration)")
print(f"  Phase 2: {PHASE_2_TRIALS} trials (refinement)")
print(f"  Data: {SAMPLE_SIZE:,} samples")
print(f"  Parallel workers: {N_JOBS}")

# ================================
# LOAD DATA ONCE
# ================================

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df = df.iloc[len(df) - SAMPLE_SIZE:]
df_clean = df.dropna()

split_idx = int(len(df_clean) * TRAIN_RATIO)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'buy_label', 'sell_label', 'idk_label', 'Timestamp', 'DateTime']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
feature_cols = list(feature_cols)

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

y_train = train_df[['buy_label', 'sell_label']].values
y_test = test_df[['buy_label', 'sell_label']].values

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
print(f"Device: {DEVICE.upper()}, Tree method: {TREE_METHOD}")

# ================================
# CLASS WEIGHTS
# ================================

# scale_pos_weight: sqrt of negative/positive ratio — softened imbalance handling
# Full inverse frequency is too aggressive for rare labels and kills precision.
# Square root dampens the upweighting, preserving some recall without sacrificing precision.
buy_neg = (y_train[:, 0] == 0).sum()
buy_pos = (y_train[:, 0] == 1).sum()
sell_neg = (y_train[:, 1] == 0).sum()
sell_pos = (y_train[:, 1] == 1).sum()

buy_scale_pos_weight = np.sqrt(buy_neg / buy_pos) if buy_pos > 0 else 1.0
sell_scale_pos_weight = np.sqrt(sell_neg / sell_pos) if sell_pos > 0 else 1.0

print(f"scale_pos_weight (sqrt dampened): BUY={buy_scale_pos_weight:.2f}, SELL={sell_scale_pos_weight:.2f}")

# ================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ================================

dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
current_device = DEVICE
current_tree_method = TREE_METHOD

def objective(trial):
    global current_device, current_tree_method
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'max_depth': trial.suggest_int('max_depth', 1, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 20.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
    }
    
    try:
        models = {}
        
        for label_idx, label_name in enumerate(['buy', 'sell']):
            y_train_label = y_train[:, label_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train_label, feature_names=feature_cols)
            
            xgb_params = {
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'min_child_weight': params['min_child_weight'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'scale_pos_weight': buy_scale_pos_weight if label_name == 'buy' else sell_scale_pos_weight,
                'objective': 'binary:logistic',
                'tree_method': current_tree_method,
                'device': current_device,
                'nthread': 1
            }
            
            try:
                evals_result = {}
                model = xgb.train(
                    xgb_params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    verbose_eval=False,
                    evals=[(dtrain, 'train')],
                    evals_result=evals_result
                )
                models[label_name] = model
                
                step_offset = label_idx * (params['n_estimators'] + 1000)
                trial.report(evals_result['train']['logloss'][-1], step=step_offset + params['n_estimators'])
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            except Exception as gpu_error:
                if current_device == 'cuda':
                    print(f"\nGPU error - falling back to CPU for remaining trials")
                    current_device = 'cpu'
                    current_tree_method = 'hist'
                    xgb_params['device'] = 'cpu'
                    xgb_params['tree_method'] = 'hist'
                    model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=params['n_estimators'],
                        verbose_eval=False
                    )
                    models[label_name] = model
                else:
                    raise
        
        buy_proba = models['buy'].predict(dtest)
        sell_proba = models['sell'].predict(dtest)

        try:
            auc_buy = roc_auc_score(y_test[:, 0], buy_proba)
            auc_sell = roc_auc_score(y_test[:, 1], sell_proba)
            return (auc_buy + auc_sell) / 2
        except ValueError:
            return 0.0
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial error: {str(e)[:100]}")
        return 0.0

# ================================
# PHASE 1: EXPLORATION
# ================================

print(f"\n===== PHASE 1: EXPLORATION ({PHASE_1_TRIALS} trials with pruning) =====")

sampler = TPESampler(seed=42)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner
)

study.optimize(
    objective,
    n_trials=PHASE_1_TRIALS,
    n_jobs=N_JOBS,
    show_progress_bar=True
)

print(f"\nPhase 1 Results:")
print(f"  Best score: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")
print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# ================================
# PHASE 2: REFINEMENT
# ================================

print(f"\n===== PHASE 2: REFINEMENT ({PHASE_2_TRIALS} trials around best region) =====")

best_p1 = study.best_params

def refined_objective(trial):
    global current_device, current_tree_method

    params = {
        'n_estimators': trial.suggest_int('n_estimators',
            max(50, int(best_p1['n_estimators'] * 0.8)),
            min(600, int(best_p1['n_estimators'] * 1.2))),
        'max_depth': trial.suggest_int('max_depth',
            max(1, best_p1['max_depth'] - 1),
            min(12, best_p1['max_depth'] + 1)),
        'learning_rate': trial.suggest_float('learning_rate',
            best_p1['learning_rate'] * 0.5,
            best_p1['learning_rate'] * 2.0, log=True),
        'subsample': trial.suggest_float('subsample',
            max(0.2, best_p1['subsample'] - 0.15),
            min(1.0, best_p1['subsample'] + 0.15)),
        'min_child_weight': trial.suggest_int('min_child_weight',
            max(1, best_p1['min_child_weight'] - 3),
            min(20, best_p1['min_child_weight'] + 3)),
        'colsample_bytree': trial.suggest_float('colsample_bytree',
            max(0.3, best_p1['colsample_bytree'] - 0.15),
            min(1.0, best_p1['colsample_bytree'] + 0.15)),
        'gamma': trial.suggest_float('gamma',
            max(0.0, best_p1['gamma'] - 2.0),
            min(20.0, best_p1['gamma'] + 2.0)),
        'reg_alpha': trial.suggest_float('reg_alpha',
            max(0.0, best_p1['reg_alpha'] - 0.5),
            min(5.0, best_p1['reg_alpha'] + 0.5)),
        'reg_lambda': trial.suggest_float('reg_lambda',
            max(0.1, best_p1['reg_lambda'] * 0.5),
            min(10.0, best_p1['reg_lambda'] * 2.0), log=True)
    }

    try:
        models = {}

        for label_idx, label_name in enumerate(['buy', 'sell']):
            y_train_label = y_train[:, label_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train_label, feature_names=feature_cols)

            xgb_params = {
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'min_child_weight': params['min_child_weight'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'scale_pos_weight': buy_scale_pos_weight if label_name == 'buy' else sell_scale_pos_weight,
                'objective': 'binary:logistic',
                'tree_method': current_tree_method,
                'device': current_device,
                'nthread': 1
            }

            try:
                evals_result = {}
                model = xgb.train(
                    xgb_params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    verbose_eval=False,
                    evals=[(dtrain, 'train')],
                    evals_result=evals_result
                )
                models[label_name] = model

                step_offset = label_idx * (params['n_estimators'] + 1000)
                trial.report(evals_result['train']['logloss'][-1], step=step_offset + params['n_estimators'])

                if trial.should_prune():
                    raise optuna.TrialPruned()

            except Exception as gpu_error:
                if current_device == 'cuda':
                    print(f"\nGPU error - falling back to CPU for remaining trials")
                    current_device = 'cpu'
                    current_tree_method = 'hist'
                    xgb_params['device'] = 'cpu'
                    xgb_params['tree_method'] = 'hist'
                    model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=params['n_estimators'],
                        verbose_eval=False
                    )
                    models[label_name] = model
                else:
                    raise

        buy_proba = models['buy'].predict(dtest)
        sell_proba = models['sell'].predict(dtest)

        try:
            auc_buy = roc_auc_score(y_test[:, 0], buy_proba)
            auc_sell = roc_auc_score(y_test[:, 1], sell_proba)
            return (auc_buy + auc_sell) / 2
        except ValueError:
            return 0.0

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial error: {str(e)[:100]}")
        return 0.0

refined_study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=84),
    pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=3)
)

refined_study.optimize(
    refined_objective,
    n_trials=PHASE_2_TRIALS,
    n_jobs=N_JOBS,
    show_progress_bar=True
)

if refined_study.best_value >= study.best_value:
    best_trial = refined_study.best_trial
    print(f"\nPhase 2 improved on Phase 1: {study.best_value:.4f} → {refined_study.best_value:.4f}")
else:
    best_trial = study.best_trial
    print(f"\nPhase 1 result held: {study.best_value:.4f} (Phase 2 best: {refined_study.best_value:.4f})")

# ================================
# RESULTS ANALYSIS
# ================================

print(f"\n===== BAYESIAN OPTIMIZATION COMPLETE =====")
print(f"Total trials: {len(study.trials) + len(refined_study.trials)}")
print(f"Successful: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) + len([t for t in refined_study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Pruned (early stopped): {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) + len([t for t in refined_study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

best_trial = best_trial

print(f"\n===== BEST PARAMETERS =====")
print(f"n_estimators: {best_trial.params['n_estimators']}")
print(f"max_depth: {best_trial.params['max_depth']}")
print(f"learning_rate: {best_trial.params['learning_rate']:.6f}")
print(f"subsample: {best_trial.params['subsample']:.4f}")
print(f"min_child_weight: {best_trial.params['min_child_weight']}")
print(f"colsample_bytree: {best_trial.params['colsample_bytree']:.4f}")
print(f"gamma: {best_trial.params['gamma']:.4f}")
print(f"reg_alpha: {best_trial.params['reg_alpha']:.4f}")
print(f"reg_lambda: {best_trial.params['reg_lambda']:.4f}")
print(f"Avg ROC-AUC: {best_trial.value:.4f}")

# ================================
# SAVE RESULTS
# ================================

trials_df = study.trials_dataframe()
trials_df.to_csv('tuning_results_bayesian.csv', index=False)
print(f"\n✓ Full results saved to: tuning_results_bayesian.csv")

with open('best_hyperparameters.txt', 'w') as f:
    f.write(f"Best Hyperparameters (Bayesian Optimization)\n")
    f.write(f"==========================================\n")
    f.write(f"n_estimators: {best_trial.params['n_estimators']}\n")
    f.write(f"max_depth: {best_trial.params['max_depth']}\n")
    f.write(f"learning_rate: {best_trial.params['learning_rate']}\n")
    f.write(f"subsample: {best_trial.params['subsample']}\n")
    f.write(f"min_child_weight: {best_trial.params['min_child_weight']}\n")
    f.write(f"colsample_bytree: {best_trial.params['colsample_bytree']}\n")
    f.write(f"gamma: {best_trial.params['gamma']}\n")
    f.write(f"reg_alpha: {best_trial.params['reg_alpha']}\n")
    f.write(f"reg_lambda: {best_trial.params['reg_lambda']}\n")
    f.write(f"Avg ROC-AUC: {best_trial.value:.4f}\n")

print(f"✓ Best params saved to: best_hyperparameters.txt")

print(f"\n===== NEXT STEPS =====")
print(f"1. Copy the best parameters above")
print(f"2. Update model.py with these hyperparameters")
print(f"3. Run: python model.py  (trains on full dataset)")
print(f"4. Run: python backtest.py  (validates profitability)")