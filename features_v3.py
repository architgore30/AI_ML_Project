"""
Feature Engineering v3 for 5-Class Entry-State Modeling

Purpose:
    Build a curated, leakage-safe feature set aligned to the multiclass labels
    from labelling_v2.py. This version emphasizes state, slope, exhaustion,
    chop detection, and higher-timeframe context.

Output:
    features_v3.csv with:
    - safe market + label columns from labeled_data_v2.csv
    - engineered features tailored to the 5-class label space
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ================================
# CONFIGURATION
# ================================

INPUT_PATH = "labeled_data_v2.csv"
OUTPUT_PATH = "features_v3.csv"
SHOW_PROGRESS = True
HTF_RULES = [("5min", "htf_5m"), ("15min", "htf_15m"), ("1h", "htf_1h")]
DROP_LABEL_DIAGNOSTICS = True

BASE_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "DateTime",
    "label_available",
    "state_class",
    "state_label",
    "no_trade_label",
    "trend_up_label",
    "trend_down_label",
    "reversal_up_label",
    "reversal_down_label",
]

LABEL_DIAGNOSTIC_COLUMNS = [
    "entry_price",
    "atr_at_label",
    "trend_score",
    "context_state",
    "up_target_hit_bar",
    "up_stop_hit_bar",
    "down_target_hit_bar",
    "down_stop_hit_bar",
    "horizon_return_atr",
    "favorable_fraction_up",
    "favorable_fraction_down",
    "first_touch_side",
    "up_opportunity",
    "down_opportunity",
]

INTERNAL_COLUMNS = ["_original_order", "_dt_utc"]

STAGE_NAMES = [
    "Load data",
    "Normalize datetime",
    "Select safe base columns",
    "Core trend/state features",
    "Slope/delta features",
    "Candle structure features",
    "Exhaustion/chop features",
    "Volume features",
    "Time features",
    "Build 5m HTF",
    "Build 15m HTF",
    "Build 1h HTF",
    "Merge HTF",
    "Optimize dtypes",
    "Diagnostics",
    "Write CSV",
]


# ================================
# PROGRESS TRACKING
# ================================

class StageProgress:
    def __init__(self, stage_names, enabled=True):
        self.enabled = enabled
        self._bar = tqdm(total=len(stage_names), dynamic_ncols=True, disable=not enabled)

    def start(self, stage_name):
        if self.enabled:
            self._bar.set_description(stage_name)

    def done(self):
        if self.enabled:
            self._bar.update(1)

    def close(self):
        if self.enabled:
            self._bar.close()


# ================================
# HELPER FUNCTIONS
# ================================

def safe_divide(numerator, denominator, fill_value=np.nan):
    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        denominator = denominator.replace(0, np.nan)
    else:
        denominator = np.where(denominator == 0, np.nan, denominator)

    result = numerator / denominator

    if not pd.isna(fill_value):
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.fillna(fill_value)
        else:
            result = np.where(np.isnan(result), fill_value, result)

    return result


def wilder_smoothing(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_rsi_wilder(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = wilder_smoothing(gains, period)
    avg_loss = wilder_smoothing(losses, period)
    rs = safe_divide(avg_gain, avg_loss)

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss.ne(0), 100.0)
    rsi = rsi.where(avg_gain.ne(0) | avg_loss.ne(0), 50.0)
    return rsi


def compute_true_range(high, low, close):
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def compute_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    tr = compute_true_range(high, low, close)
    atr = wilder_smoothing(tr, period)
    plus_di = 100 * safe_divide(wilder_smoothing(plus_dm, period), atr)
    minus_di = 100 * safe_divide(wilder_smoothing(minus_dm, period), atr)
    dx = 100 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = wilder_smoothing(dx, period)

    return atr, adx, plus_di, minus_di


def normalize_datetime_columns(df):
    working = df.copy()

    if "DateTime" in working.columns:
        parsed_datetime = pd.to_datetime(working["DateTime"], errors="coerce", utc=True)
    else:
        parsed_datetime = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]")

    if "Timestamp" in working.columns:
        timestamp_datetime = pd.to_datetime(working["Timestamp"], unit="s", errors="coerce", utc=True)
        parsed_datetime = parsed_datetime.fillna(timestamp_datetime)

    if parsed_datetime.isna().any():
        missing_rows = int(parsed_datetime.isna().sum())
        raise ValueError(f"Unable to parse timestamps for {missing_rows} rows.")

    dt_utc = parsed_datetime.dt.tz_convert("UTC").dt.tz_localize(None)
    working["_dt_utc"] = dt_utc

    if "DateTime" not in working.columns:
        working["DateTime"] = dt_utc.dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        missing_datetime = working["DateTime"].isna() | (working["DateTime"].astype(str).str.strip() == "")
        if missing_datetime.any():
            working.loc[missing_datetime, "DateTime"] = dt_utc.loc[missing_datetime].dt.strftime("%Y-%m-%d %H:%M:%S")

    return working


def build_higher_timeframe_features(source_df, rule, prefix, include_trend_delta=False):
    htf = (
        source_df.set_index("_dt_utc")[["Open", "High", "Low", "Close", "Volume"]]
        .resample(rule, label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    ema_20 = htf["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    trend_ema20 = safe_divide(htf["Close"], ema_20) - 1

    htf_features = pd.DataFrame(index=htf.index)
    htf_features[f"{prefix}_trend_ema20"] = trend_ema20
    htf_features[f"{prefix}_rsi_14"] = compute_rsi_wilder(htf["Close"], period=14)
    htf_features[f"{prefix}_roc_3"] = htf["Close"].pct_change(3)

    if include_trend_delta:
        htf_features[f"{prefix}_trend_delta_1"] = trend_ema20.diff(1)

    htf_features = htf_features.shift(1).reset_index()
    return htf_features


def optimize_feature_dtypes(result_df, feature_columns):
    for column in feature_columns:
        series = result_df[column]

        if pd.api.types.is_bool_dtype(series):
            result_df[column] = series.astype(np.int8)
        elif pd.api.types.is_integer_dtype(series):
            result_df[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            result_df[column] = pd.to_numeric(series, downcast="float")

    return result_df


def build_feature_dataframe(df, show_progress=False, progress=None):
    owns_progress = progress is None
    if progress is None:
        progress = StageProgress(STAGE_NAMES[1:-1], enabled=show_progress)

    dropped_label_diagnostics = []

    try:
        progress.start("Normalize datetime")
        working = df.copy()
        working["_original_order"] = np.arange(len(working), dtype=np.int64)

        if DROP_LABEL_DIAGNOSTICS:
            dropped_label_diagnostics = [column for column in LABEL_DIAGNOSTIC_COLUMNS if column in working.columns]
            if dropped_label_diagnostics:
                working = working.drop(columns=dropped_label_diagnostics)

        working = normalize_datetime_columns(working)
        working = working.sort_values(["_dt_utc", "_original_order"]).reset_index(drop=True)
        progress.done()

        progress.start("Select safe base columns")
        missing_base_columns = [column for column in BASE_COLUMNS if column not in working.columns and column != "DateTime"]
        if missing_base_columns:
            raise ValueError(f"Missing required base columns: {missing_base_columns}")
        progress.done()

        open_s = pd.to_numeric(working["Open"], errors="coerce")
        high = pd.to_numeric(working["High"], errors="coerce")
        low = pd.to_numeric(working["Low"], errors="coerce")
        close = pd.to_numeric(working["Close"], errors="coerce")
        volume = pd.to_numeric(working["Volume"], errors="coerce")

        close_prev = close.shift(1)
        bar_range = high - low
        return_1m = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_15 = close.pct_change(15)
        ret_60 = close.pct_change(60)

        ema_5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
        ema_10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        ema_30 = close.ewm(span=30, adjust=False, min_periods=30).mean()

        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_histogram = macd_line - macd_signal

        rsi_14 = compute_rsi_wilder(close, period=14)
        bb_mid_20 = close.rolling(window=20, min_periods=20).mean()
        bb_std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
        bb_upper_20 = bb_mid_20 + 2 * bb_std_20
        bb_lower_20 = bb_mid_20 - 2 * bb_std_20
        bb_range_20 = bb_upper_20 - bb_lower_20

        atr_14, adx_14, plus_di_14, minus_di_14 = compute_adx(high, low, close, period=14)
        di_spread_14 = plus_di_14 - minus_di_14
        vol_regime_5_60 = safe_divide(
            return_1m.rolling(window=5, min_periods=5).std(ddof=0),
            return_1m.rolling(window=60, min_periods=60).std(ddof=0),
        )

        hl_range_pct = safe_divide(bar_range, close)
        hl_range_sma_5 = hl_range_pct.rolling(window=5, min_periods=5).mean()
        hl_range_sma_20 = hl_range_pct.rolling(window=20, min_periods=20).mean()

        volume_sma_5 = volume.rolling(window=5, min_periods=5).mean()
        volume_sma_20 = volume.rolling(window=20, min_periods=20).mean()
        volume_ratio_20 = safe_divide(volume, volume_sma_20)

        progress.start("Core trend/state features")
        working["return_1m"] = return_1m
        working["ret_5"] = ret_5
        working["ret_15"] = ret_15
        working["ret_60"] = ret_60
        working["close_vs_ema_20"] = safe_divide(close, ema_20) - 1
        working["close_vs_ema_30"] = safe_divide(close, ema_30) - 1
        working["ema_5_20_spread"] = safe_divide(ema_5, ema_20) - 1
        working["ema_10_30_spread"] = safe_divide(ema_10, ema_30) - 1
        working["rsi_14"] = rsi_14
        working["macd_line_12_26"] = macd_line
        working["macd_signal_9"] = macd_signal
        working["macd_histogram"] = macd_histogram
        working["bb_position_20"] = safe_divide(close - bb_lower_20, bb_range_20)
        working["bb_width_20"] = safe_divide(bb_range_20, bb_mid_20)
        working["atr_pct_14"] = safe_divide(atr_14, close)
        working["adx_14"] = adx_14
        working["di_spread_14"] = di_spread_14
        working["vol_regime_5_60"] = vol_regime_5_60
        progress.done()

        progress.start("Slope/delta features")
        working["ema_30_slope_5"] = safe_divide(ema_30, ema_30.shift(5)) - 1
        working["ema_30_slope_15"] = safe_divide(ema_30, ema_30.shift(15)) - 1
        working["ema_spread_delta_3"] = working["ema_5_20_spread"] - working["ema_5_20_spread"].shift(3)
        working["rsi_delta_3"] = rsi_14 - rsi_14.shift(3)
        working["rsi_delta_5"] = rsi_14 - rsi_14.shift(5)
        working["macd_hist_delta_3"] = macd_histogram - macd_histogram.shift(3)
        working["bb_width_delta_5"] = safe_divide(working["bb_width_20"], working["bb_width_20"].shift(5)) - 1
        working["atr_pct_delta_5"] = safe_divide(working["atr_pct_14"], working["atr_pct_14"].shift(5)) - 1
        working["adx_delta_5"] = adx_14 - adx_14.shift(5)
        working["di_spread_delta_3"] = di_spread_14 - di_spread_14.shift(3)
        working["volume_ratio_delta_3"] = volume_ratio_20 - volume_ratio_20.shift(3)
        progress.done()

        progress.start("Candle structure features")
        upper_wick = high - np.maximum(open_s, close)
        lower_wick = np.minimum(open_s, close) - low

        working["hl_range_pct"] = hl_range_pct
        working["hl_range_sma_5"] = hl_range_sma_5
        working["range_expansion_5_20"] = safe_divide(hl_range_sma_5, hl_range_sma_20) - 1
        working["gap_prev_close"] = safe_divide(open_s, close_prev) - 1
        working["close_position"] = np.where(bar_range > 0, (close - low) / bar_range, 0.5)
        working["body_pct"] = safe_divide(close, open_s) - 1
        working["body_to_range"] = safe_divide((close - open_s).abs(), bar_range)
        working["upper_wick_pct"] = safe_divide(upper_wick, close)
        working["lower_wick_pct"] = safe_divide(lower_wick, close)
        working["wick_asymmetry"] = safe_divide(upper_wick - lower_wick, bar_range, fill_value=0.0)
        working["inside_bar"] = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(np.int8)
        working["outside_bar"] = ((high >= high.shift(1)) & (low <= low.shift(1))).astype(np.int8)
        progress.done()

        progress.start("Exhaustion/chop features")
        close_delta = close.diff()
        up_moves = close_delta.gt(0)
        down_moves = close_delta.lt(0)
        trend_sign = pd.Series(np.sign(close_delta), index=working.index).replace(0, np.nan).ffill()
        reversal_points = trend_sign.ne(trend_sign.shift(1)) & trend_sign.notna()
        reversal_group = reversal_points.cumsum()

        prior_high_20 = high.shift(1).rolling(window=20, min_periods=20).max()
        prior_low_20 = low.shift(1).rolling(window=20, min_periods=20).min()
        prior_range_20 = prior_high_20 - prior_low_20
        rolling_mean_20 = close.rolling(window=20, min_periods=20).mean()
        rolling_std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
        rolling_path_20 = close.diff().abs().rolling(window=20, min_periods=20).sum()

        working["consecutive_up_closes"] = up_moves.astype(np.int32).groupby((~up_moves).cumsum()).cumsum()
        working["consecutive_down_closes"] = down_moves.astype(np.int32).groupby((~down_moves).cumsum()).cumsum()
        working["bars_since_reversal"] = (
            trend_sign.groupby(reversal_group).cumcount().where(trend_sign.notna(), 0).astype(np.int32)
        )
        working["distance_to_swing_high_20"] = safe_divide(close, prior_high_20) - 1
        working["distance_to_swing_low_20"] = safe_divide(close, prior_low_20) - 1
        working["range_position_20"] = safe_divide(close - prior_low_20, prior_range_20, fill_value=0.5)
        working["stretch_z_20"] = safe_divide(close - rolling_mean_20, rolling_std_20)
        working["directional_efficiency_20"] = safe_divide((close - close.shift(20)).abs(), rolling_path_20)
        working["swing_proximity_asymmetry"] = (
            working["distance_to_swing_high_20"] - working["distance_to_swing_low_20"]
        )
        progress.done()

        progress.start("Volume features")
        working["volume_ratio_20"] = volume_ratio_20
        working["volume_trend_5_20"] = safe_divide(volume_sma_5, volume_sma_20) - 1
        working["price_volume_divergence_20"] = close.pct_change(20) - safe_divide(
            volume - volume.shift(20),
            volume.shift(20),
        )
        working["relative_volume_1d"] = safe_divide(volume, volume.shift(1440))
        working["relative_volume_7d"] = safe_divide(volume, volume.shift(10080))
        progress.done()

        progress.start("Time features")
        hour_fraction = working["_dt_utc"].dt.hour + (working["_dt_utc"].dt.minute / 60.0)
        dow = working["_dt_utc"].dt.dayofweek

        working["hour_sin"] = np.sin(2 * np.pi * hour_fraction / 24.0)
        working["hour_cos"] = np.cos(2 * np.pi * hour_fraction / 24.0)
        working["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        working["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        working["is_weekend"] = dow.isin([5, 6]).astype(np.int8)
        progress.done()

        progress.start("Build 5m HTF")
        htf_5m = build_higher_timeframe_features(working, HTF_RULES[0][0], HTF_RULES[0][1], include_trend_delta=True)
        progress.done()

        progress.start("Build 15m HTF")
        htf_15m = build_higher_timeframe_features(working, HTF_RULES[1][0], HTF_RULES[1][1], include_trend_delta=True)
        progress.done()

        progress.start("Build 1h HTF")
        htf_1h = build_higher_timeframe_features(working, HTF_RULES[2][0], HTF_RULES[2][1], include_trend_delta=False)
        progress.done()

        progress.start("Merge HTF")
        for htf_features in [htf_5m, htf_15m, htf_1h]:
            working = pd.merge_asof(
                working,
                htf_features.sort_values("_dt_utc"),
                on="_dt_utc",
                direction="backward",
            )

        htf_trend_columns = [
            "htf_5m_trend_ema20",
            "htf_15m_trend_ema20",
            "htf_1h_trend_ema20",
        ]
        working["htf_trend_alignment"] = np.sign(working[htf_trend_columns]).sum(axis=1, min_count=1)
        progress.done()

        working = working.sort_values("_original_order").reset_index(drop=True)
        output_columns = [column for column in BASE_COLUMNS if column in working.columns]
        feature_columns = [column for column in working.columns if column not in output_columns + INTERNAL_COLUMNS]
        result_df = working[output_columns + feature_columns].copy()

        progress.start("Optimize dtypes")
        result_df = optimize_feature_dtypes(result_df, feature_columns)
        progress.done()

        progress.start("Diagnostics")
        usable_rows = int((~result_df[feature_columns].isna().any(axis=1)).sum())
        htf_coverage = {
            "5m": float(result_df["htf_5m_trend_ema20"].notna().mean()),
            "15m": float(result_df["htf_15m_trend_ema20"].notna().mean()),
            "1h": float(result_df["htf_1h_trend_ema20"].notna().mean()),
        }
        nan_counts = result_df[feature_columns].isna().sum().sort_values(ascending=False)
        nan_counts = nan_counts[nan_counts > 0]

        print("\n===== FEATURES V3 SUMMARY =====")
        print(f"Rows: {len(result_df):,}")
        print(f"Feature count: {len(feature_columns):,}")
        print(f"Usable rows after warmup: {usable_rows:,}")

        print("\n===== DROPPED LABEL DIAGNOSTIC COLUMNS =====")
        if dropped_label_diagnostics:
            for column in dropped_label_diagnostics:
                print(f"- {column}")
        else:
            print("None detected.")

        print("\n===== HTF MERGE COVERAGE =====")
        print(f"5m trend coverage:  {htf_coverage['5m']:.2%}")
        print(f"15m trend coverage: {htf_coverage['15m']:.2%}")
        print(f"1h trend coverage:  {htf_coverage['1h']:.2%}")

        print("\n===== NaN SUMMARY =====")
        if nan_counts.empty:
            print("No NaNs in engineered features.")
        else:
            display_counts = nan_counts.head(25)
            print(display_counts)
            if len(nan_counts) > 25:
                print(f"... and {len(nan_counts) - 25} more feature columns with NaNs.")
        progress.done()

        return result_df
    finally:
        if owns_progress:
            progress.close()


def run_pipeline(input_path=INPUT_PATH, output_path=OUTPUT_PATH, show_progress=SHOW_PROGRESS):
    progress = StageProgress(STAGE_NAMES, enabled=show_progress)

    try:
        progress.start("Load data")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} rows from {input_path}")
        progress.done()

        result_df = build_feature_dataframe(df=df, show_progress=False, progress=progress)

        progress.start("Write CSV")
        result_df.to_csv(output_path, index=False)
        print(f"\nFeatures saved to: {output_path}")
        progress.done()

        return result_df
    finally:
        progress.close()


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
