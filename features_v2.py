"""
Feature Engineering v2 for Bitcoin Trend Detection

Purpose:
    Build a corrected, performance-first feature set for 1-minute BTC OHLCV data.
    The pipeline is fully vectorized, keeps row alignment intact, and can
    optionally append a prefixed legacy feature block for A/B testing.

Output:
    features_v2.csv with:
    - Original base columns preserved
    - Curated v2 features
    - Optional legacy_ feature block when KEEP_LEGACY_FEATURES = True
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ================================
# CONFIGURATION
# ================================

INPUT_PATH = "labeled_data.csv"
OUTPUT_PATH = "features_v2.csv"
KEEP_LEGACY_FEATURES = True
LEGACY_FEATURE_PREFIX = "legacy_"

BASE_COLUMNS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "DateTime",
    "buy_label",
    "sell_label",
    "idk_label",
]

INTERNAL_COLUMNS = ["_original_order", "_dt_utc"]
LEGACY_MIN_WARMUP = 20


# ================================
# PROGRESS TRACKING
# ================================

def get_stage_names(keep_legacy_features):
    stages = [
        "Load data",
        "Normalize datetime",
        "Base return and trend features",
        "RSI, MACD, and Bollinger features",
        "ATR, ADX, and volatility regime",
        "Candle structure features",
        "Volume features",
        "Exhaustion and range features",
        "Time features",
        "Build 5m HTF features",
        "Build 15m HTF features",
        "Build 1h HTF features",
        "Merge HTF features",
    ]

    if keep_legacy_features:
        stages.extend(
            [
                "Legacy base and moving average features",
                "Legacy volume features",
                "Legacy momentum and oscillator features",
                "Legacy technical indicator features",
            ]
        )

    stages.extend(
        [
            "Optimize dtypes",
            "Diagnostics",
            "Write CSV",
        ]
    )
    return stages


class StageProgress:
    def __init__(self, stage_names, enabled=True):
        self.stage_names = list(stage_names)
        self.enabled = enabled
        self._bar = tqdm(total=len(self.stage_names), dynamic_ncols=True, disable=not enabled)

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


def compute_rsi_simple(close, period):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    window = max(period - 1, 1)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
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


def build_higher_timeframe_features(source_df, rule, prefix):
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
    htf_features = pd.DataFrame(index=htf.index)
    htf_features[f"{prefix}_trend_ema20"] = safe_divide(htf["Close"], ema_20) - 1
    htf_features[f"{prefix}_rsi_14"] = compute_rsi_wilder(htf["Close"], period=14)
    htf_features[f"{prefix}_roc_3"] = htf["Close"].pct_change(3)

    htf_features = htf_features.shift(1).reset_index()
    return htf_features


def compute_legacy_features(price_df, prefix):
    legacy = pd.DataFrame(index=price_df.index)

    open_s = price_df["Open"]
    high = price_df["High"]
    low = price_df["Low"]
    close = price_df["Close"]
    volume = price_df["Volume"]
    close_prev = close.shift(1)
    return_1m = close.pct_change(1)
    hl_range = safe_divide(high - low, close)
    bar_range = high - low
    tr = compute_true_range(high, low, close)

    ma_5 = close.rolling(window=5, min_periods=5).mean()
    ma_10 = close.rolling(window=10, min_periods=10).mean()
    ma_20 = close.rolling(window=20, min_periods=20).mean()
    vol_sma_5 = volume.rolling(window=5, min_periods=5).mean()
    volume_delta = volume.diff()
    volume_delta_pct = safe_divide(volume_delta, volume.shift(1))

    legacy[f"{prefix}return_1m"] = return_1m
    legacy[f"{prefix}return_sma_5"] = return_1m.rolling(window=4, min_periods=4).mean()
    legacy[f"{prefix}return_sma_10"] = return_1m.rolling(window=9, min_periods=9).mean()
    legacy[f"{prefix}hl_range"] = hl_range
    legacy[f"{prefix}hl_range_sma"] = hl_range.rolling(window=5, min_periods=5).mean()
    legacy[f"{prefix}close_position"] = np.where(bar_range > 0, (close - low) / bar_range, 0.5)
    legacy[f"{prefix}oc_ratio"] = safe_divide(close - open_s, open_s)
    legacy[f"{prefix}ma_5"] = ma_5
    legacy[f"{prefix}ma_10"] = ma_10
    legacy[f"{prefix}ma_20"] = ma_20
    legacy[f"{prefix}price_above_ma5"] = safe_divide(close - ma_5, ma_5)
    legacy[f"{prefix}price_above_ma10"] = safe_divide(close - ma_10, ma_10)
    legacy[f"{prefix}roc_5"] = close.pct_change(5)
    legacy[f"{prefix}roc_10"] = close.pct_change(10)
    legacy[f"{prefix}volatility_5"] = return_1m.rolling(window=5, min_periods=5).std(ddof=0)
    legacy[f"{prefix}volatility_10"] = return_1m.rolling(window=10, min_periods=10).std(ddof=0)
    legacy[f"{prefix}volume_sma"] = vol_sma_5
    legacy[f"{prefix}volume_ratio"] = safe_divide(volume, vol_sma_5)
    legacy[f"{prefix}volume_delta"] = volume_delta
    legacy[f"{prefix}volume_delta_pct"] = volume_delta_pct
    legacy[f"{prefix}volume_delta_sma_5"] = volume_delta.rolling(window=5, min_periods=5).mean()
    legacy[f"{prefix}volume_delta_sma_10"] = volume_delta.rolling(window=10, min_periods=10).mean()
    legacy[f"{prefix}volume_delta_std_5"] = volume_delta.rolling(window=5, min_periods=5).std(ddof=0)
    price_dir = pd.Series(
        np.where(close_prev.isna(), np.nan, np.where(close >= close_prev, 1.0, -1.0)),
        index=close.index,
    )
    legacy[f"{prefix}volume_delta_signal"] = volume_delta_pct * price_dir
    legacy[f"{prefix}hsl_5"] = hl_range.rolling(window=5, min_periods=5).max()

    legacy[f"{prefix}rsi_5"] = compute_rsi_simple(close, period=5)
    legacy[f"{prefix}rsi_10"] = compute_rsi_simple(close, period=10)
    legacy[f"{prefix}rsi_14_simple"] = compute_rsi_simple(close, period=14)

    sma_12 = close.rolling(window=12, min_periods=12).mean()
    sma_26 = close.rolling(window=26, min_periods=26).mean()
    legacy_macd = sma_12 - sma_26
    legacy_macd_signal = legacy_macd.rolling(window=9, min_periods=9).mean()
    legacy[f"{prefix}macd_12_26_sma"] = legacy_macd
    legacy[f"{prefix}macd_signal_9_sma"] = legacy_macd_signal
    legacy[f"{prefix}macd_histogram_sma"] = legacy_macd - legacy_macd_signal

    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    legacy[f"{prefix}bb_upper_20"] = bb_upper
    legacy[f"{prefix}bb_lower_20"] = bb_lower
    legacy[f"{prefix}bb_position"] = safe_divide(close - bb_lower, bb_range)
    legacy[f"{prefix}bb_width"] = safe_divide(bb_range, bb_mid)

    legacy_atr = tr.rolling(window=14, min_periods=14).mean()
    legacy[f"{prefix}atr_14"] = legacy_atr

    highest_14 = high.rolling(window=14, min_periods=14).max()
    lowest_14 = low.rolling(window=14, min_periods=14).min()
    stoch_range = highest_14 - lowest_14
    legacy_k = 100 * safe_divide(close - lowest_14, stoch_range, fill_value=50.0)
    legacy[f"{prefix}stoch_k_14"] = legacy_k
    legacy[f"{prefix}stoch_d_3"] = legacy_k.rolling(window=3, min_periods=3).mean()

    typical_price = (high + low + close) / 3
    tp_sma_20 = typical_price.rolling(window=20, min_periods=20).mean()
    tp_mad_approx = (typical_price - tp_sma_20).abs().rolling(window=20, min_periods=20).mean()
    legacy[f"{prefix}cci_20"] = safe_divide(typical_price - tp_sma_20, 0.015 * tp_mad_approx)

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
    legacy[f"{prefix}di_plus_14"] = 100 * safe_divide(
        plus_dm.rolling(window=14, min_periods=14).mean(),
        legacy_atr,
    )
    legacy[f"{prefix}di_minus_14"] = 100 * safe_divide(
        minus_dm.rolling(window=14, min_periods=14).mean(),
        legacy_atr,
    )

    warmup_rows = min(LEGACY_MIN_WARMUP, len(legacy))
    if warmup_rows > 0:
        legacy.loc[legacy.index[:warmup_rows], :] = np.nan

    return legacy


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


def normalize_datetime_columns(df):
    working = df.copy()

    if "DateTime" in working.columns:
        parsed_datetime = pd.to_datetime(working["DateTime"], errors="coerce", utc=True)
    else:
        parsed_datetime = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]")

    if "Timestamp" in working.columns:
        timestamp_datetime = pd.to_datetime(working["Timestamp"], unit="s", errors="coerce", utc=True)
        parsed_datetime = parsed_datetime.fillna(timestamp_datetime)
    else:
        timestamp_datetime = pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]")

    if parsed_datetime.isna().any():
        missing_rows = parsed_datetime.isna().sum()
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


def build_feature_dataframe(df, keep_legacy_features=True, show_progress=False, progress=None):
    missing_columns = [column for column in BASE_COLUMNS if column not in df.columns and column != "DateTime"]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    owns_progress = progress is None
    if progress is None:
        stage_names = get_stage_names(keep_legacy_features)[1:-1]
        progress = StageProgress(stage_names, enabled=show_progress)

    try:
        progress.start("Normalize datetime")
        working = df.copy()
        working["_original_order"] = np.arange(len(working), dtype=np.int64)
        working = normalize_datetime_columns(working)
        working = working.sort_values(["_dt_utc", "_original_order"]).reset_index(drop=True)
        progress.done()

        open_s = pd.to_numeric(working["Open"], errors="coerce")
        high = pd.to_numeric(working["High"], errors="coerce")
        low = pd.to_numeric(working["Low"], errors="coerce")
        close = pd.to_numeric(working["Close"], errors="coerce")
        volume = pd.to_numeric(working["Volume"], errors="coerce")

        bar_range = high - low
        close_prev = close.shift(1)
        return_1m = close.pct_change(1)
        ema_5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

        progress.start("Base return and trend features")
        working["return_1m"] = return_1m
        working["roc_5"] = close.pct_change(5)
        working["price_vs_ema_20"] = safe_divide(close, ema_20) - 1
        working["ema_5_20_spread"] = safe_divide(ema_5, ema_20) - 1
        progress.done()

        progress.start("RSI, MACD, and Bollinger features")
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        bb_mid = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = bb_upper - bb_lower

        working["rsi_14"] = compute_rsi_wilder(close, period=14)
        working["macd_line_12_26"] = macd_line
        working["macd_signal_9"] = macd_signal
        working["macd_histogram"] = macd_line - macd_signal
        working["bb_position_20"] = safe_divide(close - bb_lower, bb_range)
        working["bb_width_20"] = safe_divide(bb_range, bb_mid)
        progress.done()

        progress.start("ATR, ADX, and volatility regime")
        atr_14, adx_14, _, _ = compute_adx(high, low, close, period=14)
        realized_vol_5 = return_1m.rolling(window=5, min_periods=5).std(ddof=0)
        realized_vol_60 = return_1m.rolling(window=60, min_periods=60).std(ddof=0)

        working["atr_pct_14"] = safe_divide(atr_14, close)
        working["adx_14"] = adx_14
        working["vol_regime_5_60"] = safe_divide(realized_vol_5, realized_vol_60)
        progress.done()

        progress.start("Candle structure features")
        upper_wick = high - np.maximum(open_s, close)
        lower_wick = np.minimum(open_s, close) - low

        working["hl_range_pct"] = safe_divide(bar_range, close)
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

        progress.start("Volume features")
        volume_sma_5 = volume.rolling(window=5, min_periods=5).mean()
        volume_sma_20 = volume.rolling(window=20, min_periods=20).mean()

        working["volume_ratio_20"] = safe_divide(volume, volume_sma_20)
        working["volume_trend_5_20"] = safe_divide(volume_sma_5, volume_sma_20) - 1
        working["price_volume_divergence_20"] = close.pct_change(20) - safe_divide(volume - volume.shift(20), volume.shift(20))
        working["relative_volume_1d"] = safe_divide(volume, volume.shift(1440))
        working["relative_volume_7d"] = safe_divide(volume, volume.shift(10080))
        progress.done()

        progress.start("Exhaustion and range features")
        close_delta = close.diff()
        up_moves = close_delta.gt(0)
        down_moves = close_delta.lt(0)
        trend_sign = pd.Series(np.sign(close_delta), index=working.index).replace(0, np.nan).ffill()
        reversal = trend_sign.ne(trend_sign.shift(1)) & trend_sign.notna()
        reversal_group = reversal.cumsum()
        prior_high_20 = high.shift(1).rolling(window=20, min_periods=20).max()
        prior_low_20 = low.shift(1).rolling(window=20, min_periods=20).min()
        prior_range_20 = prior_high_20 - prior_low_20

        working["consecutive_up_closes"] = up_moves.astype(np.int32).groupby((~up_moves).cumsum()).cumsum()
        working["consecutive_down_closes"] = down_moves.astype(np.int32).groupby((~down_moves).cumsum()).cumsum()
        working["bars_since_reversal"] = (
            trend_sign.groupby(reversal_group).cumcount().where(trend_sign.notna(), 0).astype(np.int32)
        )
        working["distance_to_swing_high_20"] = safe_divide(close, prior_high_20) - 1
        working["distance_to_swing_low_20"] = safe_divide(close, prior_low_20) - 1
        working["range_position_20"] = safe_divide(close - prior_low_20, prior_range_20, fill_value=0.5)
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

        progress.start("Build 5m HTF features")
        htf_5m = build_higher_timeframe_features(working, "5min", "htf_5m")
        progress.done()

        progress.start("Build 15m HTF features")
        htf_15m = build_higher_timeframe_features(working, "15min", "htf_15m")
        progress.done()

        progress.start("Build 1h HTF features")
        htf_1h = build_higher_timeframe_features(working, "1h", "htf_1h")
        progress.done()

        progress.start("Merge HTF features")
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

        if keep_legacy_features:
            progress.start("Legacy base and moving average features")
            legacy_features = compute_legacy_features(working[["Open", "High", "Low", "Close", "Volume"]], LEGACY_FEATURE_PREFIX)
            legacy_base_columns = [
                f"{LEGACY_FEATURE_PREFIX}return_1m",
                f"{LEGACY_FEATURE_PREFIX}return_sma_5",
                f"{LEGACY_FEATURE_PREFIX}return_sma_10",
                f"{LEGACY_FEATURE_PREFIX}hl_range",
                f"{LEGACY_FEATURE_PREFIX}hl_range_sma",
                f"{LEGACY_FEATURE_PREFIX}close_position",
                f"{LEGACY_FEATURE_PREFIX}oc_ratio",
                f"{LEGACY_FEATURE_PREFIX}ma_5",
                f"{LEGACY_FEATURE_PREFIX}ma_10",
                f"{LEGACY_FEATURE_PREFIX}ma_20",
                f"{LEGACY_FEATURE_PREFIX}price_above_ma5",
                f"{LEGACY_FEATURE_PREFIX}price_above_ma10",
                f"{LEGACY_FEATURE_PREFIX}roc_5",
                f"{LEGACY_FEATURE_PREFIX}roc_10",
                f"{LEGACY_FEATURE_PREFIX}volatility_5",
                f"{LEGACY_FEATURE_PREFIX}volatility_10",
                f"{LEGACY_FEATURE_PREFIX}hsl_5",
            ]
            for column in legacy_base_columns:
                working[column] = legacy_features[column]
            progress.done()

            progress.start("Legacy volume features")
            legacy_volume_columns = [
                f"{LEGACY_FEATURE_PREFIX}volume_sma",
                f"{LEGACY_FEATURE_PREFIX}volume_ratio",
                f"{LEGACY_FEATURE_PREFIX}volume_delta",
                f"{LEGACY_FEATURE_PREFIX}volume_delta_pct",
                f"{LEGACY_FEATURE_PREFIX}volume_delta_sma_5",
                f"{LEGACY_FEATURE_PREFIX}volume_delta_sma_10",
                f"{LEGACY_FEATURE_PREFIX}volume_delta_std_5",
                f"{LEGACY_FEATURE_PREFIX}volume_delta_signal",
            ]
            for column in legacy_volume_columns:
                working[column] = legacy_features[column]
            progress.done()

            progress.start("Legacy momentum and oscillator features")
            legacy_momentum_columns = [
                f"{LEGACY_FEATURE_PREFIX}rsi_5",
                f"{LEGACY_FEATURE_PREFIX}rsi_10",
                f"{LEGACY_FEATURE_PREFIX}rsi_14_simple",
                f"{LEGACY_FEATURE_PREFIX}macd_12_26_sma",
                f"{LEGACY_FEATURE_PREFIX}macd_signal_9_sma",
                f"{LEGACY_FEATURE_PREFIX}macd_histogram_sma",
                f"{LEGACY_FEATURE_PREFIX}stoch_k_14",
                f"{LEGACY_FEATURE_PREFIX}stoch_d_3",
            ]
            for column in legacy_momentum_columns:
                working[column] = legacy_features[column]
            progress.done()

            progress.start("Legacy technical indicator features")
            legacy_technical_columns = [
                f"{LEGACY_FEATURE_PREFIX}bb_upper_20",
                f"{LEGACY_FEATURE_PREFIX}bb_lower_20",
                f"{LEGACY_FEATURE_PREFIX}bb_position",
                f"{LEGACY_FEATURE_PREFIX}bb_width",
                f"{LEGACY_FEATURE_PREFIX}atr_14",
                f"{LEGACY_FEATURE_PREFIX}cci_20",
                f"{LEGACY_FEATURE_PREFIX}di_plus_14",
                f"{LEGACY_FEATURE_PREFIX}di_minus_14",
            ]
            for column in legacy_technical_columns:
                working[column] = legacy_features[column]
            progress.done()

        working = working.sort_values("_original_order").reset_index(drop=True)
        output_columns = [column for column in BASE_COLUMNS if column in working.columns]
        feature_columns = [column for column in working.columns if column not in output_columns + INTERNAL_COLUMNS]
        result_df = working[output_columns + feature_columns].copy()

        progress.start("Optimize dtypes")
        result_df = optimize_feature_dtypes(result_df, feature_columns)
        progress.done()

        htf_coverage = {
            "5m": float(result_df["htf_5m_trend_ema20"].notna().mean()),
            "15m": float(result_df["htf_15m_trend_ema20"].notna().mean()),
            "1h": float(result_df["htf_1h_trend_ema20"].notna().mean()),
        }

        progress.start("Diagnostics")
        nan_counts = result_df[feature_columns].isna().sum().sort_values(ascending=False)
        nan_counts = nan_counts[nan_counts > 0]
        usable_rows = int((~result_df[feature_columns].isna().any(axis=1)).sum())
        memory_mb = result_df.memory_usage(deep=True).sum() / 1024**2

        print("\n===== FEATURES V2 SUMMARY =====")
        print(f"Rows: {len(result_df):,}")
        print(f"Feature count: {len(feature_columns):,}")
        print(f"KEEP_LEGACY_FEATURES: {keep_legacy_features}")
        print(f"Usable rows after warmup: {usable_rows:,}")
        print(f"Memory usage: {memory_mb:.2f} MB")

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


def run_pipeline(input_path=INPUT_PATH, output_path=OUTPUT_PATH, keep_legacy_features=KEEP_LEGACY_FEATURES):
    stage_names = get_stage_names(keep_legacy_features)
    progress = StageProgress(stage_names, enabled=True)

    try:
        progress.start("Load data")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} rows from {input_path}")
        progress.done()

        result_df = build_feature_dataframe(
            df=df,
            keep_legacy_features=keep_legacy_features,
            show_progress=False,
            progress=progress,
        )

        progress.start("Write CSV")
        result_df.to_csv(output_path, index=False)
        print(f"\nFeatures saved to: {output_path}")
        progress.done()

        return result_df
    finally:
        progress.close()


def main():
    run_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        keep_legacy_features=KEEP_LEGACY_FEATURES,
    )


if __name__ == "__main__":
    main()
