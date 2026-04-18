import joblib

import pipeline_config as config
from pipeline_utils import (
    apply_architecture_calibration,
    artifact_paths,
    build_prediction_frame,
    collect_model_outputs,
    compute_architecture_scores,
    ensure_directories,
    get_feature_columns,
    get_model_spec_map,
    load_features_dataframe,
    load_json,
    split_dataframe,
    generate_actions,
)


def main():
    ensure_directories()

    df = load_features_dataframe(config.RUN_PROFILE)
    splits = split_dataframe(df)
    feature_cols = joblib.load(artifact_paths()["feature_columns"])
    thresholds = load_json(artifact_paths()["thresholds"])

    models = {}
    calibrators = {}
    for spec in config.MODEL_SPECS:
        spec_name = spec["name"]
        models[spec_name] = joblib.load(config.ARTIFACTS_DIR / f"{spec_name}.joblib")

    if config.ARCHITECTURE_NAME == "multiclass_baseline":
        calibrator_keys = ["p_no_trade", "p_buy", "p_exit"]
    else:
        calibrator_keys = [spec["name"] for spec in config.MODEL_SPECS]

    for key in calibrator_keys:
        calibrators[key] = joblib.load(config.ARTIFACTS_DIR / f"calibrator_{key}.joblib")

    raw_outputs = collect_model_outputs(models, splits["test"], feature_cols)
    calibrated_outputs = apply_architecture_calibration(raw_outputs, calibrators)
    score_df = compute_architecture_scores(calibrated_outputs)
    prediction_df = build_prediction_frame(splits["test"], score_df)
    decision_df = generate_actions(prediction_df, thresholds)

    output_path = config.OUTPUTS_DIR / "predictions_decisions.csv"
    decision_df.to_csv(output_path, index=False)
    print(f"Saved decisions to {output_path}")


if __name__ == "__main__":
    main()
