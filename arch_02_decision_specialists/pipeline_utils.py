from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import deque
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, log_loss, roc_auc_score
from tqdm import tqdm

import pipeline_config as config

ACTION_HOLD = np.int8(0)
ACTION_BUY = np.int8(1)
ACTION_EXIT = np.int8(2)
ACTION_LABELS = np.array(["HOLD", "BUY", "EXIT"], dtype=object)


def _resolve_parallel_jobs(n_jobs: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    requested_jobs = int(n_jobs or 1)
    return max(1, min(requested_jobs, task_count))


def _run_tasks_with_optional_parallel(
    tasks,
    worker,
    n_jobs: int | None = None,
    show_progress: bool = False,
    desc: str = "Tasks",
    prefer: str = "threads",
):
    task_list = list(tasks)
    if not task_list:
        return []

    parallel_jobs = _resolve_parallel_jobs(n_jobs, len(task_list))
    if parallel_jobs == 1:
        iterator = task_list
        if show_progress and len(task_list) > 1:
            iterator = tqdm(iterator, total=len(task_list), desc=desc, leave=False, dynamic_ncols=True)
        return [worker(task) for task in iterator]

    if show_progress:
        print(f"{desc}: {len(task_list)} task(s) across {parallel_jobs} jobs")

    return joblib.Parallel(n_jobs=parallel_jobs, prefer=prefer)(
        joblib.delayed(worker)(task) for task in task_list
    )


class IdentityCalibrator:
    def fit(self, scores, y_true):
        return self

    def predict(self, scores):
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.model = None

    def fit(self, scores, y_true):
        scores = np.asarray(scores, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int8)

        if len(np.unique(y_true)) < 2:
            self.model = IdentityCalibrator()
            self.model.fit(scores, y_true)
            return self

        clipped = np.clip(scores, 1e-6, 1 - 1e-6)

        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(clipped, y_true)
        else:
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(clipped.reshape(-1, 1), y_true)

        return self

    def predict(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)

        if self.model is None:
            return clipped
        if isinstance(self.model, IsotonicRegression):
            return np.clip(self.model.predict(clipped), 0.0, 1.0)
        if isinstance(self.model, IdentityCalibrator):
            return self.model.predict(clipped)
        return np.clip(self.model.predict_proba(clipped.reshape(-1, 1))[:, 1], 0.0, 1.0)


def ensure_directories():
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload):
    ensure_directories()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_divide(numerator, denominator):
    denominator = np.where(np.asarray(denominator) == 0, np.nan, denominator)
    return numerator / denominator


def run_local_script(script_name: str):
    subprocess.run([sys.executable, script_name], cwd=config.BASE_DIR, check=True)


def ensure_prepared_data():
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {config.DATASET_PATH}")

    if not config.LABELED_PATH.exists():
        run_local_script("labelling_v2.py")

    if not config.FEATURES_PATH.exists():
        run_local_script("features_v3.py")


def read_csv_tail(path: Path, row_limit: int, chunksize: int):
    tail_chunks = deque()
    rows_kept = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        tail_chunks.append(chunk)
        rows_kept += len(chunk)

        while tail_chunks and rows_kept - len(tail_chunks[0]) >= row_limit:
            rows_kept -= len(tail_chunks.popleft())

    if not tail_chunks:
        return pd.DataFrame()

    combined = pd.concat(list(tail_chunks), ignore_index=True)
    if len(combined) > row_limit:
        combined = combined.iloc[-row_limit:].reset_index(drop=True)
    return combined


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "macd_line_pct_close" not in working.columns and {"macd_line_12_26", "Close"}.issubset(working.columns):
        close = working["Close"].replace(0, np.nan)
        atr_abs = (working["atr_pct_14"] * close) if "atr_pct_14" in working.columns else np.nan
        working["macd_line_pct_close"] = safe_divide(working["macd_line_12_26"], close)
        working["macd_signal_pct_close"] = safe_divide(working["macd_signal_9"], close)
        working["macd_hist_pct_close"] = safe_divide(working["macd_histogram"], close)
        working["macd_hist_atr"] = safe_divide(working["macd_histogram"], atr_abs)
        working["log_close"] = np.where(working["Close"] > 0, np.log(working["Close"]), np.nan)
        working["log_volume"] = np.log1p(working["Volume"].clip(lower=0))

    return working


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    float_cols = working.select_dtypes(include=["float64"]).columns
    int_cols = working.select_dtypes(include=["int64"]).columns

    for column in float_cols:
        working[column] = pd.to_numeric(working[column], downcast="float")
    for column in int_cols:
        working[column] = pd.to_numeric(working[column], downcast="integer")

    if "state_label" in working.columns:
        working["state_label"] = working["state_label"].astype("category")

    return working


def load_features_dataframe(profile: str | None = None, row_limit: int | None = None) -> pd.DataFrame:
    ensure_prepared_data()

    profile = profile or config.RUN_PROFILE
    if row_limit is not None:
        df = read_csv_tail(config.FEATURES_PATH, row_limit, config.CSV_CHUNK_SIZE)
    elif profile == "smoke":
        df = read_csv_tail(config.FEATURES_PATH, config.SMOKE_ROW_LIMIT, config.CSV_CHUNK_SIZE)
    else:
        df = pd.read_csv(config.FEATURES_PATH)

    if df.empty:
        raise ValueError(f"No rows loaded from {config.FEATURES_PATH}")

    df = add_normalized_columns(df)
    df = df[df["label_available"] == 1].copy()
    df = optimize_dataframe_memory(df)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows remain after filtering label_available and dropping NaNs.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = set(config.RAW_MARKET_COLUMNS + config.LABEL_COLUMNS)
    return [column for column in df.columns if column not in exclude_cols]


def get_tuning_row_limit(profile: str | None = None) -> int | None:
    profile = profile or config.RUN_PROFILE
    if profile == "smoke":
        return config.TUNING_ROW_LIMIT_SMOKE
    return config.TUNING_ROW_LIMIT_FULL


def split_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * config.TRAIN_RATIO)
    validation_start = min(n_rows, train_end + config.PURGE_GAP_BARS)
    validation_end = min(n_rows, validation_start + int(n_rows * config.VALIDATION_RATIO))
    calibration_start = min(n_rows, validation_end + config.PURGE_GAP_BARS)
    calibration_end = min(n_rows, calibration_start + int(n_rows * config.CALIBRATION_RATIO))
    test_start = min(n_rows, calibration_end + config.PURGE_GAP_BARS)

    splits = {
        "train": df.iloc[:train_end].reset_index(drop=True),
        "validation": df.iloc[validation_start:validation_end].reset_index(drop=True),
        "calibration": df.iloc[calibration_start:calibration_end].reset_index(drop=True),
        "test": df.iloc[test_start:].reset_index(drop=True),
    }

    for split_name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(f"Split '{split_name}' is empty. Adjust smoke row limit or split ratios.")

    return splits


def get_model_spec_map():
    return {spec["name"]: spec for spec in config.MODEL_SPECS}


def get_trials(profile: str):
    if profile == "smoke":
        return config.PHASE_1_TRIALS_SMOKE, config.PHASE_2_TRIALS_SMOKE
    return config.PHASE_1_TRIALS_FULL, config.PHASE_2_TRIALS_FULL


def get_hardware_config():
    cpu_count = psutil.cpu_count(logical=False) or 1
    n_threads = max(1, int(config.THREADS_PER_JOB))
    available_job_slots = max(1, cpu_count - int(config.RESERVED_CPU_CORES))
    n_jobs = max(1, min(int(config.TARGET_PARALLEL_JOBS), available_job_slots // n_threads))
    device = "cpu"
    tree_method = "hist"

    try:
        probe = xgb.DMatrix(np.random.rand(64, 4), label=np.random.randint(0, 2, 64))
        xgb.train(
            {
                "tree_method": "gpu_hist",
                "device": "cuda",
                "objective": "binary:logistic",
            },
            probe,
            num_boost_round=1,
            verbose_eval=False,
        )
        device = "cuda"
        tree_method = "gpu_hist"
    except Exception:
        device = "cpu"
        tree_method = "hist"

    return {
        "cpu_count": cpu_count,
        "available_job_slots": available_job_slots,
        "n_jobs": n_jobs,
        "n_threads": n_threads,
        "device": device,
        "tree_method": tree_method,
    }


def filter_df_for_spec(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    eligible_classes = spec.get("eligible_classes")
    if not eligible_classes:
        return df.copy()
    return df[df["state_class"].isin(eligible_classes)].reset_index(drop=True)


def get_targets_for_spec(df: pd.DataFrame, spec: dict) -> np.ndarray:
    if spec["task"] == "multiclass":
        return df["state_class"].astype(np.int32).to_numpy()
    positive_classes = set(spec["positive_classes"])
    return df["state_class"].isin(positive_classes).astype(np.int8).to_numpy()


def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=np.float32, copy=False)


def get_spec_eval_metric(spec: dict) -> str:
    if "eval_metric" in spec:
        return str(spec["eval_metric"])
    if spec["task"] == "multiclass":
        return "mlogloss"
    return "logloss"


def clamp_threshold(value: float) -> float:
    return float(min(config.THRESHOLD_MAX, max(config.THRESHOLD_MIN, value)))


def build_local_threshold_grid(center: float) -> list[float]:
    return sorted({clamp_threshold(center + offset) for offset in config.SPECIALIST_THRESHOLD_OFFSETS})


def describe_class_weighting(y: np.ndarray, spec: dict) -> str:
    if not config.USE_CLASS_WEIGHTS:
        return "class_weights=off"

    if spec["task"] == "multiclass":
        counts = np.bincount(y.astype(np.int32), minlength=spec["num_class"]).astype(int)
        total = int(counts.sum())
        return f"class_weights=on(multiclass power={config.MULTICLASS_CLASS_WEIGHT_POWER} counts={counts.tolist()} total={total})"

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    spw = compute_binary_scale_pos_weight(y)
    return (
        "class_weights=on("
        f"binary power={config.BINARY_SCALE_POS_WEIGHT_POWER} neg={negatives} pos={positives} scale_pos_weight={spw:.3f}"
        ")"
    )


def build_training_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    feature_cols: list[str],
    params: dict,
    weights: np.ndarray | None = None,
    ref = None
):
    tree_method = params.get("tree_method", "hist")

    if hasattr(xgb, "QuantileDMatrix") and tree_method in {"hist", "gpu_hist"}:
        return xgb.QuantileDMatrix(features, label=labels, weight=weights, feature_names=feature_cols, ref=ref)

    return xgb.DMatrix(features, label=labels, weight=weights, feature_names=feature_cols)


def compute_binary_scale_pos_weight(y: np.ndarray) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives <= 0:
        return 1.0
    ratio = float(negatives / positives)
    return float(ratio**float(config.BINARY_SCALE_POS_WEIGHT_POWER))


def compute_multiclass_weights(y: np.ndarray, num_class: int) -> np.ndarray:
    class_counts = np.bincount(y, minlength=num_class).astype(np.float64)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    total = class_counts.sum()
    class_weights = total / (num_class * class_counts)
    class_weights = class_weights ** float(config.MULTICLASS_CLASS_WEIGHT_POWER)
    return class_weights[y]


def get_booster_best_iteration(booster, requested_rounds: int) -> int:
    attr = booster.attr("best_iteration")
    if attr is not None:
        return max(0, int(attr))

    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is not None:
        return max(0, int(best_iteration))

    return max(0, int(requested_rounds) - 1)


def get_booster_best_score(booster):
    attr = booster.attr("best_score")
    if attr is not None:
        try:
            return float(attr)
        except ValueError:
            return attr

    best_score = getattr(booster, "best_score", None)
    if best_score is None:
        return None
    try:
        return float(best_score)
    except (TypeError, ValueError):
        return best_score


def build_eval_history_frame(evals_result: dict) -> pd.DataFrame:
    if not evals_result:
        return pd.DataFrame()

    max_rounds = 0
    for dataset_metrics in evals_result.values():
        for values in dataset_metrics.values():
            max_rounds = max(max_rounds, len(values))

    rows = []
    for iteration in range(max_rounds):
        row = {"iteration": iteration + 1}
        for dataset_name, dataset_metrics in evals_result.items():
            for metric_name, values in dataset_metrics.items():
                row[f"{dataset_name}_{metric_name}"] = float(values[iteration]) if iteration < len(values) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def save_eval_history_artifacts(spec_name: str, training_info: dict) -> dict[str, str]:
    history_csv = config.ARTIFACTS_DIR / f"eval_history_{spec_name}.csv"
    info_json = config.ARTIFACTS_DIR / f"training_info_{spec_name}.json"

    history_df = build_eval_history_frame(training_info.get("evals_result", {}))
    if not history_df.empty:
        history_df.to_csv(history_csv, index=False)

    save_json(info_json, training_info)
    return {"history_csv": str(history_csv), "info_json": str(info_json)}


def build_default_params(spec: dict, hardware: dict, num_boost_round: int | None = None) -> tuple[dict, int]:
    params = {
        "max_depth": config.DEFAULT_MAX_DEPTH,
        "learning_rate": config.DEFAULT_LEARNING_RATE,
        "subsample": config.DEFAULT_SUBSAMPLE,
        "min_child_weight": config.DEFAULT_MIN_CHILD_WEIGHT,
        "colsample_bytree": config.DEFAULT_COLSAMPLE_BYTREE,
        "gamma": config.DEFAULT_GAMMA,
        "reg_alpha": config.DEFAULT_REG_ALPHA,
        "reg_lambda": config.DEFAULT_REG_LAMBDA,
        "tree_method": hardware["tree_method"],
        "device": hardware["device"],
        "nthread": hardware["n_threads"],
        "eval_metric": get_spec_eval_metric(spec),
    }

    if spec["task"] == "multiclass":
        params["objective"] = spec["objective"]
        params["num_class"] = spec["num_class"]
    else:
        params["objective"] = spec["objective"]

    rounds = int(num_boost_round or config.DEFAULT_BOOST_ROUNDS)
    return params, rounds


def build_trial_params(trial, spec: dict, hardware: dict, profile: str) -> tuple[dict, int]:
    if profile == "smoke":
        n_estimators_low = config.PHASE_1_N_ESTIMATORS_LOW_SMOKE
        n_estimators_high = config.PHASE_1_N_ESTIMATORS_HIGH_SMOKE
    else:
        n_estimators_low = config.PHASE_1_N_ESTIMATORS_LOW_FULL
        n_estimators_high = config.PHASE_1_N_ESTIMATORS_HIGH_FULL

    params = {
        "max_depth": trial.suggest_int("max_depth", config.PHASE_1_MAX_DEPTH_LOW, config.PHASE_1_MAX_DEPTH_HIGH),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.PHASE_1_LEARNING_RATE_LOW,
            config.PHASE_1_LEARNING_RATE_HIGH,
            log=True,
        ),
        "subsample": trial.suggest_float("subsample", config.PHASE_1_SUBSAMPLE_LOW, config.PHASE_1_SUBSAMPLE_HIGH),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            config.PHASE_1_MIN_CHILD_WEIGHT_LOW,
            config.PHASE_1_MIN_CHILD_WEIGHT_HIGH,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            config.PHASE_1_COLSAMPLE_BYTREE_LOW,
            config.PHASE_1_COLSAMPLE_BYTREE_HIGH,
        ),
        "gamma": trial.suggest_float("gamma", config.PHASE_1_GAMMA_LOW, config.PHASE_1_GAMMA_HIGH),
        "reg_alpha": trial.suggest_float("reg_alpha", config.PHASE_1_REG_ALPHA_LOW, config.PHASE_1_REG_ALPHA_HIGH),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            config.PHASE_1_REG_LAMBDA_LOW,
            config.PHASE_1_REG_LAMBDA_HIGH,
            log=True,
        ),
        "tree_method": hardware["tree_method"],
        "device": hardware["device"],
        "nthread": hardware["n_threads"],
        "eval_metric": get_spec_eval_metric(spec),
    }

    if spec["task"] == "multiclass":
        params["objective"] = spec["objective"]
        params["num_class"] = spec["num_class"]
    else:
        params["objective"] = spec["objective"]

    num_boost_round = trial.suggest_int("n_estimators", n_estimators_low, n_estimators_high)
    return params, num_boost_round


def _progress_callbacks(total_rounds: int, desc: str, position: int | None = None):
    tqdm_kwargs = {
        "total": total_rounds,
        "desc": desc,
        "leave": False,
        "dynamic_ncols": True,
    }
    if position is not None:
        tqdm_kwargs["position"] = position
    pbar = tqdm(**tqdm_kwargs)

    class TqdmCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False

    return pbar, TqdmCallback()


def train_booster(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    spec: dict,
    params: dict,
    num_boost_round: int,
    hardware: dict,
    validation_df: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_position: int | None = None,
):
    params = params.copy()
    fit_df = filter_df_for_spec(train_df, spec)
    X_train = get_feature_matrix(fit_df, feature_cols)
    y_train = get_targets_for_spec(fit_df, spec)

    valid_fit_df = None
    X_valid = None
    y_valid = None
    dvalid = None
    if validation_df is not None:
        valid_fit_df = filter_df_for_spec(validation_df, spec)
        if not valid_fit_df.empty:
            X_valid = get_feature_matrix(valid_fit_df, feature_cols)
            y_valid = get_targets_for_spec(valid_fit_df, spec)

    if spec["task"] == "multiclass":
        weights = None
        if config.USE_CLASS_WEIGHTS:
            weights = compute_multiclass_weights(y_train, spec["num_class"])
        dtrain = build_training_matrix(X_train, y_train, feature_cols, params, weights=weights)
        if X_valid is not None and y_valid is not None:
            valid_weights = None
            if config.USE_CLASS_WEIGHTS:
                valid_weights = compute_multiclass_weights(y_valid, spec["num_class"])
            dvalid = build_training_matrix(X_valid, y_valid, feature_cols, params, weights=valid_weights, ref=dtrain)
    else:
        if config.USE_CLASS_WEIGHTS:
            params["scale_pos_weight"] = compute_binary_scale_pos_weight(y_train)
        dtrain = build_training_matrix(X_train, y_train, feature_cols, params)
        if X_valid is not None and y_valid is not None:
            dvalid = build_training_matrix(X_valid, y_valid, feature_cols, params, ref=dtrain)

    callbacks = []
    pbar = None
    if show_progress:
        pbar, callback = _progress_callbacks(
            num_boost_round,
            progress_desc or f"  {spec['name']}",
            position=progress_position,
        )
        callbacks = [callback]

    evals = [(dtrain, "train")]
    if dvalid is not None:
        evals.append((dvalid, "validation"))

    evals_result = {}

    try:
        if show_progress and progress_position is None:
            print(f"  {spec['name']}: {describe_class_weighting(y_train, spec)}")
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
            callbacks=callbacks,
            verbose_eval=False,
        )
    finally:
        if pbar is not None:
            pbar.close()

    best_iteration = get_booster_best_iteration(booster, num_boost_round)
    used_rounds = best_iteration + 1
    best_score = get_booster_best_score(booster)
    booster.set_attr(
        best_iteration=str(best_iteration),
        used_rounds=str(used_rounds),
        requested_rounds=str(int(num_boost_round)),
        eval_metric=str(get_spec_eval_metric(spec)),
    )
    if best_score is not None:
        booster.set_attr(best_score=str(best_score))

    training_info = {
        "spec_name": spec["name"],
        "task": spec["task"],
        "eval_metric": get_spec_eval_metric(spec),
        "requested_rounds": int(num_boost_round),
        "best_iteration": int(best_iteration),
        "used_rounds": int(used_rounds),
        "best_score": best_score,
        "early_stopping_rounds": int(early_stopping_rounds) if early_stopping_rounds is not None else None,
        "train_rows": int(len(fit_df)),
        "validation_rows": int(len(valid_fit_df)) if valid_fit_df is not None else 0,
        "evals_result": evals_result,
    }

    return booster, training_info


def predict_probabilities(booster, df: pd.DataFrame, feature_cols: list[str], spec: dict):
    dmatrix = xgb.DMatrix(get_feature_matrix(df, feature_cols), feature_names=feature_cols)
    used_rounds = booster.attr("used_rounds")
    if used_rounds is not None:
        preds = booster.predict(dmatrix, iteration_range=(0, int(used_rounds)))
    else:
        preds = booster.predict(dmatrix)
    if spec["task"] == "multiclass":
        preds = preds.reshape(len(df), spec["num_class"])
    return preds


def score_for_tuning(spec: dict, y_true: np.ndarray, predictions) -> float:
    metric = spec.get("tuning_metric", "roc_auc")
    if spec["task"] == "multiclass":
        if metric == "accuracy":
            return float((np.argmax(predictions, axis=1) == y_true).mean())
        return -float(log_loss(y_true, predictions, labels=list(range(spec["num_class"]))))

    y_true = np.asarray(y_true, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.0

    if metric == "roc_auc":
        return float(roc_auc_score(y_true, predictions))
    if metric == "fbeta":
        thresholds = np.linspace(0.15, 0.85, 15)
        beta = spec.get("beta", 1.0)
        best = 0.0
        for threshold in thresholds:
            y_pred = (predictions >= threshold).astype(np.int8)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best:
                best = float(score)
        return best
    if metric == "neg_logloss":
        return -float(log_loss(y_true, predictions))

    return float(roc_auc_score(y_true, predictions))


def evaluate_predictions(spec: dict, y_true: np.ndarray, predictions) -> dict:
    metrics = {}

    if spec["task"] == "multiclass":
        metrics["logloss"] = float(log_loss(y_true, predictions, labels=list(range(spec["num_class"]))))
        metrics["accuracy"] = float((np.argmax(predictions, axis=1) == y_true).mean())
        return metrics

    y_true = np.asarray(y_true, dtype=np.int8)
    predictions = np.clip(np.asarray(predictions, dtype=np.float64), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, predictions))
    metrics["logloss"] = float(log_loss(y_true, predictions))

    threshold = 0.5
    y_pred = (predictions >= threshold).astype(np.int8)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["precision_at_0_5"] = float(precision)
    metrics["recall_at_0_5"] = float(recall)
    metrics["fbeta_at_0_5"] = float(
        fbeta_score(y_true, y_pred, beta=spec.get("beta", 1.0), zero_division=0)
    )
    return metrics


def collect_model_outputs(
    models: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_jobs: int | None = None,
    show_progress: bool = False,
) -> dict:
    spec_map = get_model_spec_map()

    def worker(task):
        name, booster = task
        return name, predict_probabilities(booster, df, feature_cols, spec_map[name])

    results = _run_tasks_with_optional_parallel(
        models.items(),
        worker,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc="Model outputs",
    )
    return dict(results)


def fit_architecture_calibrators(
    raw_outputs: dict,
    calibration_df: pd.DataFrame,
    n_jobs: int | None = None,
    show_progress: bool = False,
) -> dict:
    calibrators = {}
    arch = config.ARCHITECTURE_NAME

    def fit_task(task):
        name, scores, y_true = task
        calibrator = ProbabilityCalibrator(config.CALIBRATION_METHOD)
        calibrator.fit(scores, y_true)
        return name, calibrator

    if arch == "multiclass_baseline":
        class_probs = raw_outputs["multiclass_state"]
        targets = {
            "p_no_trade": (calibration_df["state_class"] == 0).astype(np.int8).to_numpy(),
            "p_buy": calibration_df["state_class"].isin([1, 3]).astype(np.int8).to_numpy(),
            "p_exit": calibration_df["state_class"].isin([2, 4]).astype(np.int8).to_numpy(),
        }
        source_scores = {
            "p_no_trade": class_probs[:, 0],
            "p_buy": class_probs[:, 1] + class_probs[:, 3],
            "p_exit": class_probs[:, 2] + class_probs[:, 4],
        }

        tasks = [(name, source_scores[name], y_true) for name, y_true in targets.items()]
        for name, calibrator in _run_tasks_with_optional_parallel(
            tasks,
            fit_task,
            n_jobs=n_jobs,
            show_progress=show_progress,
            desc="Calibrators",
        ):
            calibrators[name] = calibrator
        return calibrators

    spec_map = get_model_spec_map()
    tasks = []
    for name, raw_scores in raw_outputs.items():
        spec = spec_map[name]
        eligible_df = filter_df_for_spec(calibration_df, spec)
        eligible_classes = spec.get("eligible_classes")
        if eligible_classes:
            eligible_mask = calibration_df["state_class"].isin(eligible_classes).to_numpy()
        else:
            eligible_mask = np.ones(len(calibration_df), dtype=bool)
        y_true = get_targets_for_spec(eligible_df, spec)
        tasks.append((name, np.asarray(raw_scores)[eligible_mask], y_true))

    for name, calibrator in _run_tasks_with_optional_parallel(
        tasks,
        fit_task,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc="Calibrators",
    ):
        calibrators[name] = calibrator

    return calibrators


def apply_architecture_calibration(
    raw_outputs: dict,
    calibrators: dict,
    n_jobs: int | None = None,
    show_progress: bool = False,
) -> dict:
    arch = config.ARCHITECTURE_NAME

    if arch == "multiclass_baseline":
        class_probs = raw_outputs["multiclass_state"]
        return {
            "multiclass_state": class_probs,
            "p_no_trade": calibrators["p_no_trade"].predict(class_probs[:, 0]),
            "p_buy": calibrators["p_buy"].predict(class_probs[:, 1] + class_probs[:, 3]),
            "p_exit": calibrators["p_exit"].predict(class_probs[:, 2] + class_probs[:, 4]),
        }

    def worker(task):
        name, raw_scores = task
        return name, calibrators[name].predict(raw_scores)

    results = _run_tasks_with_optional_parallel(
        raw_outputs.items(),
        worker,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc="Apply calibration",
    )
    return dict(results)


def compute_architecture_scores(calibrated_outputs: dict) -> pd.DataFrame:
    arch = config.ARCHITECTURE_NAME

    if arch == "multiclass_baseline":
        class_probs = calibrated_outputs["multiclass_state"]
        p_no_trade = np.asarray(calibrated_outputs["p_no_trade"])
        p_buy = np.asarray(calibrated_outputs["p_buy"])
        p_exit = np.asarray(calibrated_outputs["p_exit"])

        scores = pd.DataFrame(
            {
                "p_no_trade": p_no_trade,
                "p_buy": p_buy,
                "p_exit": p_exit,
                "p_hold": np.clip(np.maximum(p_no_trade, 1 - np.maximum(p_buy, p_exit)), 0.0, 1.0),
                "score_buy_trend": class_probs[:, 1],
                "score_buy_reversal": class_probs[:, 3],
                "score_exit_trend": class_probs[:, 2],
                "score_exit_reversal": class_probs[:, 4],
            }
        )
        return scores

    if arch == "decision_specialists":
        p_no_trade = np.asarray(calibrated_outputs["no_trade_guard"])
        p_long_entry = np.asarray(calibrated_outputs["long_entry"])
        p_downside_exit = np.asarray(calibrated_outputs["downside_exit"])
        p_long_trend = np.asarray(calibrated_outputs["long_archetype"])
        p_long_reversal = 1.0 - p_long_trend
        p_downside_trend = np.asarray(calibrated_outputs["downside_archetype"])
        p_downside_reversal = 1.0 - p_downside_trend

        score_buy_trend = p_long_entry * p_long_trend
        score_buy_reversal = p_long_entry * p_long_reversal
        score_exit_trend = p_downside_exit * p_downside_trend
        score_exit_reversal = p_downside_exit * p_downside_reversal
        p_buy = np.maximum(score_buy_trend, score_buy_reversal)
        p_exit = np.maximum(score_exit_trend, score_exit_reversal)

        return pd.DataFrame(
            {
                "p_no_trade": p_no_trade,
                "p_buy": p_buy,
                "p_exit": p_exit,
                "p_hold": np.clip(np.maximum(p_no_trade, 1 - np.maximum(p_buy, p_exit)), 0.0, 1.0),
                "p_long_entry": p_long_entry,
                "p_downside_exit": p_downside_exit,
                "p_long_trend": p_long_trend,
                "p_long_reversal": p_long_reversal,
                "p_downside_trend": p_downside_trend,
                "p_downside_reversal": p_downside_reversal,
                "score_buy_trend": score_buy_trend,
                "score_buy_reversal": score_buy_reversal,
                "score_exit_trend": score_exit_trend,
                "score_exit_reversal": score_exit_reversal,
            }
        )

    p_no_trade = np.asarray(calibrated_outputs["no_trade_guard"])
    p_trend_router = np.asarray(calibrated_outputs["archetype_router"])
    p_reversal_router = 1.0 - p_trend_router
    p_trend_up = np.asarray(calibrated_outputs["trend_direction"])
    p_trend_down = 1.0 - p_trend_up
    p_reversal_up = np.asarray(calibrated_outputs["reversal_direction"])
    p_reversal_down = 1.0 - p_reversal_up

    score_buy_trend = p_trend_router * p_trend_up
    score_exit_trend = p_trend_router * p_trend_down
    score_buy_reversal = p_reversal_router * p_reversal_up
    score_exit_reversal = p_reversal_router * p_reversal_down
    p_buy = np.maximum(score_buy_trend, score_buy_reversal)
    p_exit = np.maximum(score_exit_trend, score_exit_reversal)

    return pd.DataFrame(
        {
            "p_no_trade": p_no_trade,
            "p_buy": p_buy,
            "p_exit": p_exit,
            "p_hold": np.clip(np.maximum(p_no_trade, 1 - np.maximum(p_buy, p_exit)), 0.0, 1.0),
            "p_trend_router": p_trend_router,
            "p_reversal_router": p_reversal_router,
            "p_trend_up": p_trend_up,
            "p_trend_down": p_trend_down,
            "p_reversal_up": p_reversal_up,
            "p_reversal_down": p_reversal_down,
            "score_buy_trend": score_buy_trend,
            "score_buy_reversal": score_buy_reversal,
            "score_exit_trend": score_exit_trend,
            "score_exit_reversal": score_exit_reversal,
        }
    )


def build_prediction_frame(df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    output_cols = ["Timestamp", "DateTime", "Open", "High", "Low", "Close"]
    return pd.concat([df[output_cols].reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)


def build_prediction_arrays(prediction_df: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {
        "Timestamp": prediction_df["Timestamp"].to_numpy(),
        "Open": prediction_df["Open"].to_numpy(dtype=np.float64),
        "Close": prediction_df["Close"].to_numpy(dtype=np.float64),
        "p_no_trade": prediction_df["p_no_trade"].to_numpy(dtype=np.float64),
        "p_buy": prediction_df["p_buy"].to_numpy(dtype=np.float64),
        "p_exit": prediction_df["p_exit"].to_numpy(dtype=np.float64),
    }

    if config.ARCHITECTURE_NAME == "decision_specialists":
        arrays.update(
            {
                "p_long_entry": prediction_df["p_long_entry"].to_numpy(dtype=np.float64),
                "p_downside_exit": prediction_df["p_downside_exit"].to_numpy(dtype=np.float64),
                "score_buy_trend": prediction_df["score_buy_trend"].to_numpy(dtype=np.float64),
                "score_buy_reversal": prediction_df["score_buy_reversal"].to_numpy(dtype=np.float64),
                "score_exit_trend": prediction_df["score_exit_trend"].to_numpy(dtype=np.float64),
                "score_exit_reversal": prediction_df["score_exit_reversal"].to_numpy(dtype=np.float64),
            }
        )
    elif config.ARCHITECTURE_NAME == "archetype_specialists":
        arrays.update(
            {
                "p_trend_router": prediction_df["p_trend_router"].to_numpy(dtype=np.float64),
                "p_reversal_router": prediction_df["p_reversal_router"].to_numpy(dtype=np.float64),
                "score_buy_trend": prediction_df["score_buy_trend"].to_numpy(dtype=np.float64),
                "score_buy_reversal": prediction_df["score_buy_reversal"].to_numpy(dtype=np.float64),
                "score_exit_trend": prediction_df["score_exit_trend"].to_numpy(dtype=np.float64),
                "score_exit_reversal": prediction_df["score_exit_reversal"].to_numpy(dtype=np.float64),
            }
        )

    return arrays


def generate_action_codes_from_arrays(prediction_arrays: dict[str, np.ndarray], thresholds: dict) -> np.ndarray:
    row_count = len(prediction_arrays["p_buy"])
    actions = np.full(row_count, ACTION_HOLD, dtype=np.int8)

    p_buy = prediction_arrays["p_buy"]
    p_exit = prediction_arrays["p_exit"]
    p_no_trade = prediction_arrays["p_no_trade"]

    no_trade_threshold = float(thresholds["no_trade_threshold"])
    entry_threshold = float(thresholds["entry_threshold"])
    exit_threshold = float(thresholds["exit_threshold"])

    if config.ARCHITECTURE_NAME == "decision_specialists":
        p_long_entry = prediction_arrays["p_long_entry"]
        p_downside_exit = prediction_arrays["p_downside_exit"]
        score_buy_trend = prediction_arrays["score_buy_trend"]
        score_buy_reversal = prediction_arrays["score_buy_reversal"]
        score_exit_trend = prediction_arrays["score_exit_trend"]
        score_exit_reversal = prediction_arrays["score_exit_reversal"]
        no_trade_guard_threshold = float(thresholds.get("no_trade_guard_threshold", no_trade_threshold))
        long_entry_threshold = float(thresholds.get("long_entry_threshold", entry_threshold))
        downside_exit_threshold = float(thresholds.get("downside_exit_threshold", exit_threshold))
        downside_block_threshold = float(thresholds.get("downside_block_threshold", downside_exit_threshold))
        buy_trend_threshold = float(thresholds.get("buy_trend_threshold", entry_threshold))
        buy_reversal_threshold = float(thresholds.get("buy_reversal_threshold", entry_threshold))
        exit_trend_threshold = float(thresholds.get("exit_trend_threshold", exit_threshold))
        exit_reversal_threshold = float(thresholds.get("exit_reversal_threshold", exit_threshold))
    elif config.ARCHITECTURE_NAME == "archetype_specialists":
        p_trend_router = prediction_arrays["p_trend_router"]
        p_reversal_router = prediction_arrays["p_reversal_router"]
        score_buy_trend = prediction_arrays["score_buy_trend"]
        score_buy_reversal = prediction_arrays["score_buy_reversal"]
        score_exit_trend = prediction_arrays["score_exit_trend"]
        score_exit_reversal = prediction_arrays["score_exit_reversal"]
        no_trade_guard_threshold = float(thresholds.get("no_trade_guard_threshold", no_trade_threshold))
        trend_router_threshold = float(thresholds.get("trend_router_threshold", 0.5))
        reversal_router_threshold = float(thresholds.get("reversal_router_threshold", 0.5))
        trend_up_threshold = float(thresholds.get("trend_up_threshold", entry_threshold))
        reversal_up_threshold = float(thresholds.get("reversal_up_threshold", entry_threshold))
        trend_down_threshold = float(thresholds.get("trend_down_threshold", exit_threshold))
        reversal_down_threshold = float(thresholds.get("reversal_down_threshold", exit_threshold))

    in_position = False
    entry_exec_index = None
    pending_entry_index = None
    pending_exit_index = None

    for idx in range(row_count):
        if pending_entry_index == idx:
            in_position = True
            entry_exec_index = idx
            pending_entry_index = None

        if pending_exit_index == idx:
            in_position = False
            entry_exec_index = None
            pending_exit_index = None

        if idx >= row_count - 1:
            continue

        if not in_position:
            if config.ARCHITECTURE_NAME == "decision_specialists":
                no_trade_ok = p_no_trade[idx] <= no_trade_guard_threshold
                long_entry_ok = p_long_entry[idx] >= long_entry_threshold
                downside_block = p_downside_exit[idx] >= downside_block_threshold
                trend_buy_ok = score_buy_trend[idx] >= buy_trend_threshold
                reversal_buy_ok = score_buy_reversal[idx] >= buy_reversal_threshold
                if no_trade_ok and long_entry_ok and not downside_block and (trend_buy_ok or reversal_buy_ok):
                    actions[idx] = ACTION_BUY
                    pending_entry_index = idx + 1
            elif config.ARCHITECTURE_NAME == "archetype_specialists":
                no_trade_ok = p_no_trade[idx] <= no_trade_guard_threshold
                trend_router_ok = p_trend_router[idx] >= trend_router_threshold
                reversal_router_ok = p_reversal_router[idx] >= reversal_router_threshold
                trend_buy_ok = score_buy_trend[idx] >= trend_up_threshold
                reversal_buy_ok = score_buy_reversal[idx] >= reversal_up_threshold
                if no_trade_ok and ((trend_router_ok and trend_buy_ok) or (reversal_router_ok and reversal_buy_ok)):
                    actions[idx] = ACTION_BUY
                    pending_entry_index = idx + 1
            else:
                if p_buy[idx] >= entry_threshold and p_no_trade[idx] <= no_trade_threshold and p_exit[idx] < exit_threshold:
                    actions[idx] = ACTION_BUY
                    pending_entry_index = idx + 1
        else:
            bars_held = idx - entry_exec_index if entry_exec_index is not None else 0
            if config.ARCHITECTURE_NAME == "decision_specialists":
                exit_signal = (
                    p_downside_exit[idx] >= downside_exit_threshold
                    or score_exit_trend[idx] >= exit_trend_threshold
                    or score_exit_reversal[idx] >= exit_reversal_threshold
                )
            elif config.ARCHITECTURE_NAME == "archetype_specialists":
                exit_signal = (
                    score_exit_trend[idx] >= trend_down_threshold
                    or score_exit_reversal[idx] >= reversal_down_threshold
                )
            else:
                exit_signal = p_exit[idx] >= exit_threshold

            if bars_held >= config.ENTRY_TIMEOUT_BARS or exit_signal:
                actions[idx] = ACTION_EXIT
                pending_exit_index = idx + 1

    return actions


def generate_actions(prediction_df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    output = prediction_df.copy()
    prediction_arrays = build_prediction_arrays(prediction_df)
    action_codes = generate_action_codes_from_arrays(prediction_arrays, thresholds)
    output["action"] = ACTION_LABELS[action_codes]
    return output


def simulate_backtest_from_arrays(
    prediction_arrays: dict[str, np.ndarray],
    action_codes: np.ndarray,
    include_details: bool = True,
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    prices_open = prediction_arrays["Open"]
    prices_close = prediction_arrays["Close"]
    timestamps = prediction_arrays["Timestamp"]

    cash_balance = float(config.INITIAL_BALANCE)
    btc_holdings = 0.0
    entry_exec_index = None
    entry_price = None
    pending_entry_index = None
    pending_exit_index = None
    pending_exit_reason = None

    running_max = float(config.INITIAL_BALANCE)
    max_drawdown_pct = 0.0
    trade_count = 0
    winning_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0

    equity_curve = [] if include_details else None
    trade_log = [] if include_details else None

    for idx in range(len(action_codes)):
        if pending_entry_index == idx:
            execution_price = prices_open[idx] * (1 + config.SLIPPAGE)
            btc_holdings = cash_balance / (execution_price * (1 + config.COMMISSION))
            cash_spent = btc_holdings * execution_price * (1 + config.COMMISSION)
            cash_balance -= cash_spent
            entry_exec_index = idx
            entry_price = execution_price
            pending_entry_index = None

        if pending_exit_index == idx and btc_holdings > 0 and entry_exec_index is not None:
            execution_price = prices_open[idx] * (1 - config.SLIPPAGE)
            cash_from_sale = btc_holdings * execution_price * (1 - config.COMMISSION)
            entry_cost = btc_holdings * entry_price * (1 + config.COMMISSION)
            pnl_usd = cash_from_sale - entry_cost
            pnl_pct = pnl_usd / entry_cost if entry_cost else 0.0

            trade_count += 1
            if pnl_usd > 0:
                winning_trades += 1
                gross_profit += float(pnl_usd)
            else:
                gross_loss += float(pnl_usd)

            if include_details:
                trade_log.append(
                    {
                        "entry_exec_index": entry_exec_index,
                        "exit_exec_index": idx,
                        "entry_timestamp": timestamps[entry_exec_index],
                        "exit_timestamp": timestamps[idx],
                        "entry_price": entry_price,
                        "exit_price": execution_price,
                        "holding_bars": idx - entry_exec_index,
                        "pnl_usd": pnl_usd,
                        "pnl_pct": pnl_pct,
                        "reason": pending_exit_reason or "signal_exit",
                    }
                )

            cash_balance += cash_from_sale
            btc_holdings = 0.0
            entry_exec_index = None
            entry_price = None
            pending_exit_index = None
            pending_exit_reason = None

        action_code = int(action_codes[idx])
        if btc_holdings > 0 and entry_exec_index is not None:
            bars_held = idx - entry_exec_index
            if bars_held >= config.ENTRY_TIMEOUT_BARS and idx < len(action_codes) - 1 and pending_exit_index is None:
                action_code = ACTION_EXIT
                pending_exit_reason = "timeout"

        if action_code == ACTION_BUY and btc_holdings == 0 and idx < len(action_codes) - 1:
            pending_entry_index = idx + 1

        if action_code == ACTION_EXIT and btc_holdings > 0 and idx < len(action_codes) - 1 and pending_exit_index is None:
            pending_exit_index = idx + 1
            pending_exit_reason = pending_exit_reason or "model_exit"

        equity = cash_balance + btc_holdings * prices_close[idx]
        if equity > running_max:
            running_max = float(equity)
        elif running_max > 0:
            drawdown_pct = ((equity / running_max) - 1.0) * 100.0
            if drawdown_pct < max_drawdown_pct:
                max_drawdown_pct = float(drawdown_pct)

        if include_details:
            equity_curve.append(
                {
                    "Timestamp": timestamps[idx],
                    "equity": equity,
                    "action": ACTION_LABELS[action_codes[idx]],
                }
            )

    if btc_holdings > 0 and entry_exec_index is not None:
        execution_price = prices_close[-1] * (1 - config.SLIPPAGE)
        cash_from_sale = btc_holdings * execution_price * (1 - config.COMMISSION)
        entry_cost = btc_holdings * entry_price * (1 + config.COMMISSION)
        pnl_usd = cash_from_sale - entry_cost
        pnl_pct = pnl_usd / entry_cost if entry_cost else 0.0

        trade_count += 1
        if pnl_usd > 0:
            winning_trades += 1
            gross_profit += float(pnl_usd)
        else:
            gross_loss += float(pnl_usd)

        if include_details:
            trade_log.append(
                {
                    "entry_exec_index": entry_exec_index,
                    "exit_exec_index": len(action_codes) - 1,
                    "entry_timestamp": timestamps[entry_exec_index],
                    "exit_timestamp": timestamps[-1],
                    "entry_price": entry_price,
                    "exit_price": execution_price,
                    "holding_bars": len(action_codes) - 1 - entry_exec_index,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                    "reason": "end_of_data",
                }
            )

        cash_balance += cash_from_sale

    final_balance = float(cash_balance)
    total_return_pct = ((final_balance / config.INITIAL_BALANCE) - 1.0) * 100.0
    win_rate = float(winning_trades / trade_count) if trade_count else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else None

    summary = {
        "architecture": config.ARCHITECTURE_NAME,
        "final_balance": final_balance,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "trade_count": int(trade_count),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }

    if not include_details:
        return summary, None, None

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trade_log)
    return summary, trades_df, equity_df

def simulate_backtest(prediction_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    prediction_arrays = build_prediction_arrays(prediction_df)
    action_values = prediction_df["action"].to_numpy()
    action_codes = np.full(len(action_values), ACTION_HOLD, dtype=np.int8)
    action_codes[action_values == "BUY"] = ACTION_BUY
    action_codes[action_values == "EXIT"] = ACTION_EXIT
    summary, trades_df, equity_df = simulate_backtest_from_arrays(
        prediction_arrays,
        action_codes,
        include_details=True,
    )
    return summary, trades_df, equity_df


def threshold_objective(summary: dict) -> float:
    penalty = config.THRESHOLD_OBJECTIVE_DRAWDOWN_WEIGHT * abs(summary["max_drawdown_pct"])
    return float(summary["total_return_pct"] - penalty + 0.01 * summary["trade_count"])


def search_base_thresholds(prediction_arrays: dict[str, np.ndarray], show_progress: bool = False) -> tuple[dict, dict]:
    best_thresholds = None
    best_summary = None
    best_objective = -np.inf

    candidates = [
        {
            "no_trade_threshold": float(no_trade_threshold),
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
        }
        for no_trade_threshold in config.NO_TRADE_THRESHOLD_GRID
        for entry_threshold in config.ENTRY_THRESHOLD_GRID
        for exit_threshold in config.EXIT_THRESHOLD_GRID
    ]

    iterator = candidates
    if show_progress:
        iterator = tqdm(candidates, total=len(candidates), desc="Base thresholds", leave=False, dynamic_ncols=True)

    for thresholds in iterator:
        action_codes = generate_action_codes_from_arrays(prediction_arrays, thresholds)
        summary, _, _ = simulate_backtest_from_arrays(prediction_arrays, action_codes, include_details=False)
        objective = threshold_objective(summary)

        if objective > best_objective:
            best_objective = objective
            best_thresholds = thresholds
            best_summary = summary

    return best_thresholds, best_summary


def search_specialist_thresholds(
    prediction_arrays: dict[str, np.ndarray],
    base_thresholds: dict,
    show_progress: bool = False,
) -> tuple[dict, dict]:
    best_thresholds = None
    best_summary = None
    best_objective = -np.inf

    if config.ARCHITECTURE_NAME == "decision_specialists":
        grids = {
            "no_trade_guard_threshold": build_local_threshold_grid(base_thresholds["no_trade_threshold"]),
            "long_entry_threshold": build_local_threshold_grid(base_thresholds["entry_threshold"]),
            "buy_trend_threshold": build_local_threshold_grid(base_thresholds["entry_threshold"]),
            "buy_reversal_threshold": build_local_threshold_grid(base_thresholds["entry_threshold"]),
            "downside_exit_threshold": build_local_threshold_grid(base_thresholds["exit_threshold"]),
            "exit_trend_threshold": build_local_threshold_grid(base_thresholds["exit_threshold"]),
            "exit_reversal_threshold": build_local_threshold_grid(base_thresholds["exit_threshold"]),
        }
        keys = list(grids.keys())
        candidates = [
            {
                **base_thresholds,
                **dict(zip(keys, values)),
                "downside_block_threshold": dict(zip(keys, values))["downside_exit_threshold"],
            }
            for values in product(*(grids[key] for key in keys))
        ]
        iterator = candidates
        if show_progress:
            iterator = tqdm(
                candidates,
                total=len(candidates),
                desc="Specialist thresholds",
                leave=False,
                dynamic_ncols=True,
            )
        for thresholds in iterator:
            action_codes = generate_action_codes_from_arrays(prediction_arrays, thresholds)
            summary, _, _ = simulate_backtest_from_arrays(prediction_arrays, action_codes, include_details=False)
            objective = threshold_objective(summary)
            if objective > best_objective:
                best_objective = objective
                best_thresholds = thresholds
                best_summary = summary
        return best_thresholds, best_summary

    if config.ARCHITECTURE_NAME == "archetype_specialists":
        grids = {
            "no_trade_guard_threshold": build_local_threshold_grid(base_thresholds["no_trade_threshold"]),
            "trend_router_threshold": build_local_threshold_grid(0.5),
            "reversal_router_threshold": build_local_threshold_grid(0.5),
            "trend_up_threshold": build_local_threshold_grid(base_thresholds["entry_threshold"]),
            "reversal_up_threshold": build_local_threshold_grid(base_thresholds["entry_threshold"]),
            "trend_down_threshold": build_local_threshold_grid(base_thresholds["exit_threshold"]),
            "reversal_down_threshold": build_local_threshold_grid(base_thresholds["exit_threshold"]),
        }
        keys = list(grids.keys())
        candidates = [
            {
                **base_thresholds,
                **dict(zip(keys, values)),
            }
            for values in product(*(grids[key] for key in keys))
        ]
        iterator = candidates
        if show_progress:
            iterator = tqdm(
                candidates,
                total=len(candidates),
                desc="Specialist thresholds",
                leave=False,
                dynamic_ncols=True,
            )
        for thresholds in iterator:
            action_codes = generate_action_codes_from_arrays(prediction_arrays, thresholds)
            summary, _, _ = simulate_backtest_from_arrays(prediction_arrays, action_codes, include_details=False)
            objective = threshold_objective(summary)
            if objective > best_objective:
                best_objective = objective
                best_thresholds = thresholds
                best_summary = summary
        return best_thresholds, best_summary

    return base_thresholds, None


def search_thresholds(calibration_prediction_df: pd.DataFrame, show_progress: bool = False) -> tuple[dict, dict]:
    prediction_arrays = build_prediction_arrays(calibration_prediction_df)
    base_thresholds, base_summary = search_base_thresholds(prediction_arrays, show_progress=show_progress)

    if config.ARCHITECTURE_NAME == "multiclass_baseline":
        return base_thresholds, base_summary

    specialist_thresholds, specialist_summary = search_specialist_thresholds(
        prediction_arrays,
        base_thresholds,
        show_progress=show_progress,
    )
    if specialist_summary is None:
        return base_thresholds, base_summary

    base_objective = threshold_objective(base_summary)
    specialist_objective = threshold_objective(specialist_summary)
    if specialist_objective >= base_objective:
        return specialist_thresholds, specialist_summary
    return base_thresholds, base_summary


def get_truth_columns_for_architecture(df: pd.DataFrame) -> dict:
    return {
        "y_no_trade": (df["state_class"] == 0).astype(np.int8).to_numpy(),
        "y_buy": df["state_class"].isin([1, 3]).astype(np.int8).to_numpy(),
        "y_exit": df["state_class"].isin([2, 4]).astype(np.int8).to_numpy(),
    }


def artifact_paths():
    return {
        "best_params": config.ARTIFACTS_DIR / "best_params.json",
        "feature_columns": config.ARTIFACTS_DIR / "feature_columns.joblib",
        "split_metadata": config.ARTIFACTS_DIR / "split_metadata.json",
        "thresholds": config.ARTIFACTS_DIR / "thresholds.json",
        "metrics": config.ARTIFACTS_DIR / "metrics.json",
        "training_summary": config.ARTIFACTS_DIR / "training_summary.json",
    }
