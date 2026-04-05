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
from sklearn.metrics import precision_recall_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================
# HARDWARE DETECTION & STRATEGY
# ================================

cpu_count = psutil.cpu_count(logical=False)
is_weak_hardware = cpu_count is not None and cpu_count <= 4

print(f"Detected {cpu_count} physical CPU cores")
if is_weak_hardware:
    print("✓ Weak hardware detected (<=4 cores) - using Bayesian optimization")
    N_JOBS = 9
else:
    print("✓ Standard/strong hardware - using full optimization")
    N_JOBS = 9

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

DATA_PATH = "features_v2.csv"
TRAIN_RATIO = 0.8
SAMPLE_SIZE = 500_000
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
print(f"Buy samples: {df.buy_label.sum()}/{len(df)}")
print(f"Sell samples: {df.sell_label.sum()}/{len(df)}")
print(f"Buy samples (test): {y_test[:, 0].sum()}/{len(y_test)}")
print(f"Sell samples (test): {y_test[:, 1].sum()}/{len(y_test)}")

# ================================
# CLASS WEIGHTS
# ================================

# scale_pos_weight: unbiased negative/positive ratio
# This uses the raw inverse-frequency class weight without dampening.
buy_neg = (y_train[:, 0] == 0).sum()
buy_pos = (y_train[:, 0] == 1).sum()
sell_neg = (y_train[:, 1] == 0).sum()
sell_pos = (y_train[:, 1] == 1).sum()

buy_scale_pos_weight = (buy_neg / buy_pos) if buy_pos > 0 else 1.0
sell_scale_pos_weight = (sell_neg / sell_pos) if sell_pos > 0 else 1.0

print(f"scale_pos_weight (raw ratio): BUY={buy_scale_pos_weight:.2f}, SELL={sell_scale_pos_weight:.2f}")

# ================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ================================

dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
current_device = DEVICE
current_tree_method = TREE_METHOD


def bounded_int_window(center, lower_bound, upper_bound, delta):
    low = max(lower_bound, int(center - delta))
    high = min(upper_bound, int(center + delta))
    if low > high:
        low = high = min(max(int(center), lower_bound), upper_bound)
    return low, high


def bounded_float_window(center, lower_bound, upper_bound, delta=None, scale=None):
    if scale is not None:
        low = max(lower_bound, center * scale[0])
        high = min(upper_bound, center * scale[1])
    else:
        low = max(lower_bound, center - delta)
        high = min(upper_bound, center + delta)

    if low > high:
        clamped = min(max(center, lower_bound), upper_bound)
        low = high = clamped

    return low, high

def make_objective(label_idx, label_name, scale_pos_weight):
    def objective(trial):
        global current_device, current_tree_method

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 600),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 35),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
            'gamma': trial.suggest_float('gamma', 0, 25.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 15.0, log=True)
        }

        try:
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
                'scale_pos_weight': scale_pos_weight,
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

                trial.report(evals_result['train']['logloss'][-1], step=params['n_estimators'])

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
                else:
                    raise

            proba = model.predict(dtest)
            precision, recall, _ = precision_recall_curve(y_test[:, label_idx], proba)
            mask = recall >= 0.005
            return float(precision[mask].max()) if mask.any() else 0.0

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial error: {str(e)[:100]}")
            return 0.0
    return objective


def make_refined_objective(best_p1, label_idx, label_name, scale_pos_weight):
    def refined_objective(trial):
        global current_device, current_tree_method

        n_estimators_low, n_estimators_high = bounded_int_window(best_p1['n_estimators'], 50, 600, 0.2 * best_p1['n_estimators'])
        max_depth_low, max_depth_high = bounded_int_window(best_p1['max_depth'], 1, 15, 1)
        learning_rate_low, learning_rate_high = bounded_float_window(best_p1['learning_rate'], 0.001, 0.1, scale=(0.5, 2.0))
        subsample_low, subsample_high = bounded_float_window(best_p1['subsample'], 0.2, 1.0, delta=0.15)
        min_child_weight_low, min_child_weight_high = bounded_int_window(best_p1['min_child_weight'], 1, 25, 3)
        colsample_low, colsample_high = bounded_float_window(best_p1['colsample_bytree'], 0.3, 1.0, delta=0.15)
        gamma_low, gamma_high = bounded_float_window(best_p1['gamma'], 0.0, 25.0, delta=2.0)
        reg_alpha_low, reg_alpha_high = bounded_float_window(best_p1['reg_alpha'], 0.0, 5.0, delta=0.5)
        reg_lambda_low, reg_lambda_high = bounded_float_window(best_p1['reg_lambda'], 0.1, 10.0, scale=(0.5, 2.0))

        params = {
            'n_estimators': trial.suggest_int('n_estimators',
                n_estimators_low,
                n_estimators_high),
            'max_depth': trial.suggest_int('max_depth',
                max_depth_low,
                max_depth_high),
            'learning_rate': trial.suggest_float('learning_rate',
                learning_rate_low,
                learning_rate_high, log=True),
            'subsample': trial.suggest_float('subsample',
                subsample_low,
                subsample_high),
            'min_child_weight': trial.suggest_int('min_child_weight',
                min_child_weight_low,
                min_child_weight_high),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                colsample_low,
                colsample_high),
            'gamma': trial.suggest_float('gamma',
                gamma_low,
                gamma_high),
            'reg_alpha': trial.suggest_float('reg_alpha',
                reg_alpha_low,
                reg_alpha_high),
            'reg_lambda': trial.suggest_float('reg_lambda',
                reg_lambda_low,
                reg_lambda_high, log=True)
        }

        try:
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
                'scale_pos_weight': scale_pos_weight,
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

                trial.report(evals_result['train']['logloss'][-1], step=params['n_estimators'])

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
                else:
                    raise

            proba = model.predict(dtest)
            precision, recall, _ = precision_recall_curve(y_test[:, label_idx], proba)
            mask = recall >= 0.005
            return float(precision[mask].max()) if mask.any() else 0.0

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial error: {str(e)[:100]}")
            return 0.0
    return refined_objective


# ================================
# TUNING LOOP — BUY THEN SELL
# ================================

for label_idx, label_name, scale_pos_weight in [
    (0, 'buy',  buy_scale_pos_weight),
    (1, 'sell', sell_scale_pos_weight),
]:
    print(f"\n{'='*60}")
    print(f"  TUNING {label_name.upper()} MODEL")
    print(f"{'='*60}")

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
        make_objective(label_idx, label_name, scale_pos_weight),
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

    refined_study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=84),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )

    refined_study.optimize(
        make_refined_objective(best_p1, label_idx, label_name, scale_pos_weight),
        n_trials=PHASE_2_TRIALS,
        n_jobs=N_JOBS,
        show_progress_bar=True
    )

    if refined_study.best_value >= study.best_value:
        best_trial = refined_study.best_trial
        print(f"\nPhase 2 improved on Phase 1: {study.best_value:.4f} -> {refined_study.best_value:.4f}")
    else:
        best_trial = study.best_trial
        print(f"\nPhase 1 result held: {study.best_value:.4f} (Phase 2 best: {refined_study.best_value:.4f})")

    # ================================
    # RESULTS ANALYSIS
    # ================================

    print(f"\n===== BAYESIAN OPTIMIZATION COMPLETE ({label_name.upper()}) =====")
    print(f"Total trials: {len(study.trials) + len(refined_study.trials)}")
    print(f"Successful: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) + len([t for t in refined_study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned (early stopped): {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]) + len([t for t in refined_study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    best_trial = best_trial

    print(f"\n===== BEST PARAMETERS ({label_name.upper()}) =====")
    print(f"n_estimators: {best_trial.params['n_estimators']}")
    print(f"max_depth: {best_trial.params['max_depth']}")
    print(f"learning_rate: {best_trial.params['learning_rate']:.6f}")
    print(f"subsample: {best_trial.params['subsample']:.4f}")
    print(f"min_child_weight: {best_trial.params['min_child_weight']}")
    print(f"colsample_bytree: {best_trial.params['colsample_bytree']:.4f}")
    print(f"gamma: {best_trial.params['gamma']:.4f}")
    print(f"reg_alpha: {best_trial.params['reg_alpha']:.4f}")
    print(f"reg_lambda: {best_trial.params['reg_lambda']:.4f}")
    print(f"Score: {best_trial.value:.4f}")

    # ================================
    # SAVE RESULTS
    # ================================

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'tuning_results_{label_name}.csv', index=False)
    print(f"\n\u2713 Full results saved to: tuning_results_{label_name}.csv")

    with open(f'best_hyperparameters_{label_name}.txt', 'w') as f:
        label_num = 1 if label_name == "buy" else 2
        f.write(f"Best Hyperparameters ({label_name.upper()} Model - Bayesian Optimization)\n")
        f.write(f"==========================================\n")
        f.write(f"n_estimators{label_num} = {best_trial.params['n_estimators']}\n")
        f.write(f"max_depth{label_num} = {best_trial.params['max_depth']}\n")
        f.write(f"learning_rate{label_num} = {best_trial.params['learning_rate']}\n")
        f.write(f"subsample{label_num} = {best_trial.params['subsample']}\n")
        f.write(f"min_child_weight{label_num} = {best_trial.params['min_child_weight']}\n")
        f.write(f"colsample_bytree{label_num} = {best_trial.params['colsample_bytree']}\n")
        f.write(f"gamma{label_num} = {best_trial.params['gamma']}\n")
        f.write(f"reg_alpha{label_num} = {best_trial.params['reg_alpha']}\n")
        f.write(f"reg_lambda{label_num} = {best_trial.params['reg_lambda']}\n")
        f.write(f"Score: {best_trial.value:.4f}\n")

    print(f"\u2713 Best params saved to: best_hyperparameters_{label_name}.txt")

print(f"\n===== NEXT STEPS =====")
print(f"1. Copy the best parameters above")
print(f"2. Update model.py with these hyperparameters")
print(f"3. Run: python model.py  (trains on full dataset)")
print(f"4. Run: python backtest.py  (validates profitability)")