import gc
from contextlib import nullcontext
from queue import Queue

import optuna
import pandas as pd
from optuna.samplers import TPESampler

import pipeline_config as config
from pipeline_utils import (
    artifact_paths,
    build_trial_params,
    ensure_directories,
    filter_df_for_spec,
    get_feature_columns,
    get_hardware_config,
    get_model_spec_map,
    get_targets_for_spec,
    get_tuning_row_limit,
    get_trials,
    load_features_dataframe,
    predict_probabilities,
    save_json,
    score_for_tuning,
    split_dataframe,
    train_booster,
)


optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_progress_slot_pool(n_slots: int) -> Queue:
    slots = Queue(maxsize=n_slots)
    for position in range(1, n_slots + 1):
        slots.put(position)
    return slots


class progress_slot:
    def __init__(self, slot_pool: Queue):
        self.slot_pool = slot_pool
        self.position = None

    def __enter__(self):
        self.position = self.slot_pool.get()
        return self.position

    def __exit__(self, exc_type, exc, tb):
        self.slot_pool.put(self.position)
        return False


def bounded_int_window(center, low, high, delta):
    left = max(low, int(center - delta))
    right = min(high, int(center + delta))
    if left > right:
        clamped = min(max(int(center), low), high)
        return clamped, clamped
    return left, right


def bounded_float_window(center, low, high, delta):
    left = max(low, float(center - delta))
    right = min(high, float(center + delta))
    if left > right:
        clamped = min(max(float(center), low), high)
        return clamped, clamped
    return left, right


def make_refined_params(best_trial, spec, hardware, profile):
    params = best_trial.params
    n_low, n_high = bounded_int_window(
        params["n_estimators"],
        config.PHASE_2_N_ESTIMATORS_HARD_LOW,
        config.PHASE_2_N_ESTIMATORS_HARD_HIGH,
        max(config.PHASE_2_N_ESTIMATORS_WINDOW_MIN, config.PHASE_2_N_ESTIMATORS_WINDOW_RATIO * params["n_estimators"]),
    )
    d_low, d_high = bounded_int_window(
        params["max_depth"],
        config.PHASE_2_MAX_DEPTH_HARD_LOW,
        config.PHASE_2_MAX_DEPTH_HARD_HIGH,
        config.PHASE_2_MAX_DEPTH_WINDOW,
    )
    lr_low, lr_high = bounded_float_window(
        params["learning_rate"],
        config.PHASE_2_LEARNING_RATE_HARD_LOW,
        config.PHASE_2_LEARNING_RATE_HARD_HIGH,
        params["learning_rate"] * config.PHASE_2_LEARNING_RATE_WINDOW_RATIO,
    )
    ss_low, ss_high = bounded_float_window(
        params["subsample"],
        config.PHASE_2_SUBSAMPLE_HARD_LOW,
        config.PHASE_2_SUBSAMPLE_HARD_HIGH,
        config.PHASE_2_SUBSAMPLE_WINDOW,
    )
    mcw_low, mcw_high = bounded_int_window(
        params["min_child_weight"],
        config.PHASE_2_MIN_CHILD_WEIGHT_HARD_LOW,
        config.PHASE_2_MIN_CHILD_WEIGHT_HARD_HIGH,
        config.PHASE_2_MIN_CHILD_WEIGHT_WINDOW,
    )
    cs_low, cs_high = bounded_float_window(
        params["colsample_bytree"],
        config.PHASE_2_COLSAMPLE_BYTREE_HARD_LOW,
        config.PHASE_2_COLSAMPLE_BYTREE_HARD_HIGH,
        config.PHASE_2_COLSAMPLE_BYTREE_WINDOW,
    )
    gamma_low, gamma_high = bounded_float_window(
        params["gamma"],
        config.PHASE_2_GAMMA_HARD_LOW,
        config.PHASE_2_GAMMA_HARD_HIGH,
        config.PHASE_2_GAMMA_WINDOW,
    )
    alpha_low, alpha_high = bounded_float_window(
        params["reg_alpha"],
        config.PHASE_2_REG_ALPHA_HARD_LOW,
        config.PHASE_2_REG_ALPHA_HARD_HIGH,
        config.PHASE_2_REG_ALPHA_WINDOW,
    )
    lambda_low, lambda_high = bounded_float_window(
        params["reg_lambda"],
        config.PHASE_2_REG_LAMBDA_HARD_LOW,
        config.PHASE_2_REG_LAMBDA_HARD_HIGH,
        params["reg_lambda"] * config.PHASE_2_REG_LAMBDA_WINDOW_RATIO,
    )

    def sampler(trial):
        refined_params = {
            "max_depth": trial.suggest_int("max_depth", d_low, d_high),
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=True),
            "subsample": trial.suggest_float("subsample", ss_low, ss_high),
            "min_child_weight": trial.suggest_int("min_child_weight", mcw_low, mcw_high),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cs_low, cs_high),
            "gamma": trial.suggest_float("gamma", gamma_low, gamma_high),
            "reg_alpha": trial.suggest_float("reg_alpha", alpha_low, alpha_high),
            "reg_lambda": trial.suggest_float("reg_lambda", lambda_low, lambda_high, log=True),
            "tree_method": hardware["tree_method"],
            "device": hardware["device"],
            "nthread": hardware["n_threads"],
            "objective": spec["objective"],
            "eval_metric": spec["eval_metric"],
        }
        if spec["task"] == "multiclass":
            refined_params["num_class"] = spec["num_class"]
        rounds = trial.suggest_int("n_estimators", n_low, n_high)
        return refined_params, rounds

    return sampler


def main():
    ensure_directories()

    tuning_row_limit = get_tuning_row_limit(config.RUN_PROFILE)
    df = load_features_dataframe(config.RUN_PROFILE, row_limit=tuning_row_limit)
    feature_cols = get_feature_columns(df)
    df = df[feature_cols + ["state_class"]].copy()
    splits = split_dataframe(df)
    hardware = get_hardware_config()
    phase_1_trials, phase_2_trials = get_trials(config.RUN_PROFILE)
    spec_map = get_model_spec_map()
    show_training_progress = True
    tuning_progress_slots = make_progress_slot_pool(hardware["n_jobs"])

    print(f"Architecture: {config.ARCHITECTURE_NAME}")
    print(f"Run profile: {config.RUN_PROFILE}")
    print(f"Tuning row limit: {tuning_row_limit if tuning_row_limit is not None else 'full dataset'}")
    print(f"Rows after prep: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(
        f"Hardware: device={hardware['device']} tree_method={hardware['tree_method']} "
        f"jobs={hardware['n_jobs']} threads_per_job={hardware['n_threads']}"
    )

    all_best_params = {}

    for spec in config.MODEL_SPECS:
        spec_name = spec["name"]
        print(f"\n===== TUNING {spec_name} =====")

        train_df = filter_df_for_spec(splits["train"], spec)
        validation_df = filter_df_for_spec(splits["validation"], spec)
        y_valid = get_targets_for_spec(validation_df, spec)

        def objective(trial, param_builder):
            params, rounds = param_builder(trial)
            booster = None
            training_info = None
            predictions = None
            slot_context = progress_slot(tuning_progress_slots) if show_training_progress else nullcontext(None)
            try:
                with slot_context as progress_position:
                    booster, training_info = train_booster(
                        train_df=train_df,
                        feature_cols=feature_cols,
                        spec=spec,
                        params=params,
                        num_boost_round=rounds,
                        hardware=hardware,
                        validation_df=validation_df,
                        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS_TUNING,
                        show_progress=show_training_progress,
                        progress_desc=f"  {spec_name} T{trial.number + 1}",
                        progress_position=progress_position,
                    )
                    predictions = predict_probabilities(booster, validation_df, feature_cols, spec)
                    score = score_for_tuning(spec, y_valid, predictions)
                    trial.set_user_attr("used_rounds", int(training_info["used_rounds"]))
                    trial.set_user_attr("best_iteration", int(training_info["best_iteration"]))
                    trial.set_user_attr("best_score", training_info["best_score"])
                    trial.set_user_attr("eval_metric", training_info["eval_metric"])
                    return score
            finally:
                del booster
                del training_info
                del predictions
                gc.collect()

        base_builder = lambda trial: build_trial_params(trial, spec, hardware, config.RUN_PROFILE)
        phase_1 = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        phase_1.optimize(
            lambda trial: objective(trial, base_builder),
            n_trials=phase_1_trials,
            n_jobs=hardware["n_jobs"],
            show_progress_bar=True,
        )

        refined_builder = make_refined_params(phase_1.best_trial, spec, hardware, config.RUN_PROFILE)
        phase_2 = optuna.create_study(direction="maximize", sampler=TPESampler(seed=84))
        phase_2.optimize(
            lambda trial: objective(trial, refined_builder),
            n_trials=phase_2_trials,
            n_jobs=hardware["n_jobs"],
            show_progress_bar=True,
        )

        best_study = phase_2 if phase_2.best_value >= phase_1.best_value else phase_1
        best_trial = best_study.best_trial
        best_params = dict(best_trial.params)
        best_params["objective"] = spec["objective"]
        if spec["task"] == "multiclass":
            best_params["num_class"] = spec["num_class"]

        all_best_params[spec_name] = {
            "params": best_params,
            "num_boost_round": int(best_trial.user_attrs.get("used_rounds", best_trial.params["n_estimators"])),
            "score": float(best_trial.value),
            "task": spec["task"],
            "best_iteration": int(best_trial.user_attrs.get("best_iteration", best_trial.params["n_estimators"] - 1)),
            "best_score": best_trial.user_attrs.get("best_score"),
            "eval_metric": best_trial.user_attrs.get("eval_metric", spec["eval_metric"]),
        }

        trials_df = pd.concat(
            [
                phase_1.trials_dataframe().assign(phase="phase_1"),
                phase_2.trials_dataframe().assign(phase="phase_2"),
            ],
            ignore_index=True,
        )
        trials_df.to_csv(config.ARTIFACTS_DIR / f"tuning_trials_{spec_name}.csv", index=False)
        print(f"Best score: {best_trial.value:.5f}")
        print(f"Best rounds used: {all_best_params[spec_name]['num_boost_round']}")

    save_json(artifact_paths()["best_params"], all_best_params)
    print(f"\nSaved best params to {artifact_paths()['best_params']}")


if __name__ == "__main__":
    main()
