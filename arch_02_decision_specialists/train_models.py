from contextlib import nullcontext
from queue import Queue

import joblib
import pandas as pd

import pipeline_config as config
from pipeline_utils import (
    apply_architecture_calibration,
    artifact_paths,
    build_default_params,
    build_prediction_frame,
    collect_model_outputs,
    compute_architecture_scores,
    ensure_directories,
    evaluate_predictions,
    filter_df_for_spec,
    fit_architecture_calibrators,
    get_feature_columns,
    get_hardware_config,
    get_targets_for_spec,
    load_features_dataframe,
    load_json,
    predict_probabilities,
    save_json,
    save_eval_history_artifacts,
    search_thresholds,
    split_dataframe,
    train_booster,
)


def make_progress_slot_pool(n_slots: int) -> Queue:
    slots = Queue(maxsize=n_slots)
    for position in range(n_slots):
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


def main():
    ensure_directories()

    df = load_features_dataframe(config.RUN_PROFILE)
    splits = split_dataframe(df)
    feature_cols = get_feature_columns(df)
    hardware = get_hardware_config()
    best_params_payload = load_json(artifact_paths()["best_params"], default={})
    final_train_df = (
        splits["train"].copy()
        if not config.TRAIN_ON_TRAIN_PLUS_VALIDATION
        else pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    )
    parallel_jobs = max(1, min(hardware["n_jobs"], len(config.MODEL_SPECS)))
    show_progress = True
    training_progress_slots = make_progress_slot_pool(parallel_jobs)

    print(f"Architecture: {config.ARCHITECTURE_NAME}")
    print(f"Training rows: {len(splits['train']):,}")
    print(f"Validation rows: {len(splits['validation']):,}")
    print(f"Calibration rows: {len(splits['calibration']):,}")
    print(f"Test rows: {len(splits['test']):,}")
    print(
        f"Hardware: device={hardware['device']} tree_method={hardware['tree_method']} "
        f"jobs={parallel_jobs} threads_per_job={hardware['n_threads']}"
    )

    models = {}
    metrics_payload = {"architecture": config.ARCHITECTURE_NAME, "models": {}}
    training_summary = {"architecture": config.ARCHITECTURE_NAME, "models": {}}

    if parallel_jobs > 1:
        spec_names = ", ".join(spec["name"] for spec in config.MODEL_SPECS)
        print(f"Training specs in parallel: {spec_names}")

    def train_spec(spec):
        spec_name = spec["name"]
        slot_context = progress_slot(training_progress_slots) if show_progress else nullcontext(None)
        with slot_context as progress_position:
            if show_progress and progress_position is None:
                print(f"\n===== TRAINING {spec_name} =====")

            saved = best_params_payload.get(spec_name)
            if saved:
                params = saved["params"].copy()
                num_boost_round = int(saved["num_boost_round"])
                params.pop("n_estimators", None)
            else:
                params, num_boost_round = build_default_params(spec, hardware)

            selection_booster, selection_info = train_booster(
                train_df=splits["train"],
                feature_cols=feature_cols,
                spec=spec,
                params=params,
                num_boost_round=num_boost_round,
                hardware=hardware,
                validation_df=splits["validation"],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS_TRAINING,
                show_progress=show_progress,
                progress_desc=f"  {spec_name} select",
                progress_position=progress_position,
            )

            validation_df = filter_df_for_spec(splits["validation"], spec)
            y_valid = get_targets_for_spec(validation_df, spec)
            validation_predictions = predict_probabilities(selection_booster, validation_df, feature_cols, spec)
            validation_metrics = evaluate_predictions(spec, y_valid, validation_predictions)
            eval_artifacts = save_eval_history_artifacts(spec_name, selection_info)

            final_rounds = int(selection_info["used_rounds"])
            final_booster, final_info = train_booster(
                train_df=final_train_df,
                feature_cols=feature_cols,
                spec=spec,
                params=params,
                num_boost_round=final_rounds,
                hardware=hardware,
                validation_df=None,
                early_stopping_rounds=None,
                show_progress=show_progress,
                progress_desc=f"  {spec_name} final ",
                progress_position=progress_position,
            )
            return {
                "spec_name": spec_name,
                "booster": final_booster,
                "validation_metrics": validation_metrics,
                "selection_info": selection_info,
                "final_info": final_info,
                "eval_artifacts": eval_artifacts,
            }

    training_results = joblib.Parallel(n_jobs=parallel_jobs, prefer="threads")(
        joblib.delayed(train_spec)(spec) for spec in config.MODEL_SPECS
    )

    for result in training_results:
        spec_name = result["spec_name"]
        models[spec_name] = result["booster"]
        metrics_payload["models"][spec_name] = {
            "validation_metrics": result["validation_metrics"],
            "selection_training": {
                "best_iteration": result["selection_info"]["best_iteration"],
                "used_rounds": result["selection_info"]["used_rounds"],
                "best_score": result["selection_info"]["best_score"],
                "eval_metric": result["selection_info"]["eval_metric"],
                "train_rows": result["selection_info"]["train_rows"],
                "validation_rows": result["selection_info"]["validation_rows"],
            },
            "final_training": {
                "used_rounds": result["final_info"]["used_rounds"],
                "requested_rounds": result["final_info"]["requested_rounds"],
                "eval_metric": result["final_info"]["eval_metric"],
                "train_rows": result["final_info"]["train_rows"],
            },
        }
        training_summary["models"][spec_name] = {
            "selection": result["selection_info"],
            "final": result["final_info"],
            "eval_artifacts": result["eval_artifacts"],
        }

        model_path = config.ARTIFACTS_DIR / f"{spec_name}.joblib"
        joblib.dump(result["booster"], model_path)
        print(f"Saved model to {model_path}")

    print("\nGenerating calibration model outputs...")
    raw_calibration_outputs = collect_model_outputs(
        models,
        splits["calibration"],
        feature_cols,
        n_jobs=parallel_jobs,
        show_progress=True,
    )
    print("Fitting calibration layers...")
    calibrators = fit_architecture_calibrators(
        raw_calibration_outputs,
        splits["calibration"],
        n_jobs=parallel_jobs,
        show_progress=True,
    )
    print("Applying calibration layers...")
    calibrated_calibration_outputs = apply_architecture_calibration(
        raw_calibration_outputs,
        calibrators,
        n_jobs=parallel_jobs,
        show_progress=True,
    )
    print("Computing calibration scores...")
    calibration_scores = compute_architecture_scores(calibrated_calibration_outputs)
    calibration_predictions = build_prediction_frame(splits["calibration"], calibration_scores)

    print("Searching thresholds...")
    thresholds, threshold_summary = search_thresholds(calibration_predictions, show_progress=True)
    metrics_payload["threshold_search"] = threshold_summary

    print("Saving calibration artifacts...")
    for calibrator_name, calibrator in calibrators.items():
        joblib.dump(calibrator, config.ARTIFACTS_DIR / f"calibrator_{calibrator_name}.joblib")

    joblib.dump(feature_cols, artifact_paths()["feature_columns"])
    save_json(artifact_paths()["thresholds"], thresholds)
    save_json(
        artifact_paths()["split_metadata"],
        {split_name: len(split_df) for split_name, split_df in splits.items()},
    )
    save_json(artifact_paths()["training_summary"], training_summary)

    print("Building summary metrics...")
    truth = splits["calibration"]
    if config.ARCHITECTURE_NAME == "multiclass_baseline":
        raw_class_probs = raw_calibration_outputs["multiclass_state"]
        raw_score_summary = {
            "p_no_trade_mean": float(raw_class_probs[:, 0].mean()),
            "p_buy_mean": float((raw_class_probs[:, 1] + raw_class_probs[:, 3]).mean()),
            "p_exit_mean": float((raw_class_probs[:, 2] + raw_class_probs[:, 4]).mean()),
        }
    else:
        raw_scores = compute_architecture_scores(raw_calibration_outputs)
        raw_score_summary = {
            "p_no_trade_mean": float(raw_scores["p_no_trade"].mean()),
            "p_buy_mean": float(raw_scores["p_buy"].mean()),
            "p_exit_mean": float(raw_scores["p_exit"].mean()),
        }

    calibrated_score_summary = {
        "p_no_trade_mean": float(calibration_scores["p_no_trade"].mean()),
        "p_buy_mean": float(calibration_scores["p_buy"].mean()),
        "p_exit_mean": float(calibration_scores["p_exit"].mean()),
    }

    metrics_payload["calibration_truth_rates"] = {
        "p_no_trade": float((truth["state_class"] == 0).mean()),
        "p_buy": float(truth["state_class"].isin([1, 3]).mean()),
        "p_exit": float(truth["state_class"].isin([2, 4]).mean()),
    }
    metrics_payload["calibration_score_summary"] = {
        "raw_means": raw_score_summary,
        "calibrated_means": calibrated_score_summary,
    }

    save_json(artifact_paths()["metrics"], metrics_payload)
    print(f"\nSaved thresholds to {artifact_paths()['thresholds']}")
    print(f"Saved metrics to {artifact_paths()['metrics']}")
    print(f"Saved training summary to {artifact_paths()['training_summary']}")


if __name__ == "__main__":
    main()
