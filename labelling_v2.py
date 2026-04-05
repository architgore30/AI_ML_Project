"""
Labeling v2 for Bitcoin Trend Detection

Purpose:
    Generate 5-class entry-state labels using path-aware future OHLC,
    next-bar-open entry, ATR-scaled targets/stops, and past-only trend context.

Output:
    labeled_data_v2.csv with:
    - original market data columns
    - one multiclass state label
    - one-hot state columns
    - optional diagnostic columns for debugging and analysis
"""

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


# ================================
# CONFIGURATION
# ================================

DATA_PATH = "dataset.csv"
OUTPUT_PATH = "labeled_data_v2.csv"

POST_2018_ONLY = True
POST_2018_START = "2018-01-01"

ATR_PERIOD = 20
TREND_EMA_PERIOD = 30
CONTEXT_RET_SHORT = 15
CONTEXT_RET_LONG = 60

HORIZON = 30
TARGET_MULT = 1.5
STOP_MULT = 1.0
MIN_HORIZON_RETURN_MULT = 0.5
MIN_FAVORABLE_OCCUPANCY = 0.55
AMBIGUOUS_REVERSAL_WINDOW = 3

CHUNK_SIZE = 200_000
INCLUDE_DIAGNOSTIC_COLUMNS = True
SHOW_PROGRESS = True

REQUIRED_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
CLASS_ID_TO_NAME = {
    -1: "unlabeled",
    0: "no_trade",
    1: "trend_up",
    2: "trend_down",
    3: "reversal_up",
    4: "reversal_down",
}
CLASS_OUTPUT_COLUMNS = [
    "no_trade_label",
    "trend_up_label",
    "trend_down_label",
    "reversal_up_label",
    "reversal_down_label",
]


# ================================
# PROGRESS TRACKING
# ================================

class StageProgress:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._bar = tqdm(total=0, dynamic_ncols=True, disable=not enabled)
        self._pending_updates = 0

    def set_total(self, total):
        if not self.enabled:
            return

        self._bar.total = total
        self._bar.refresh()
        if self._pending_updates:
            self._bar.update(self._pending_updates)
            self._pending_updates = 0

    def start(self, stage_name):
        if self.enabled:
            self._bar.set_description(stage_name)

    def done(self):
        if not self.enabled:
            return

        if self._bar.total:
            self._bar.update(1)
        else:
            self._pending_updates += 1

    def close(self):
        if self.enabled:
            self._bar.close()


# ================================
# HELPER FUNCTIONS
# ================================

def wilder_smoothing(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


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


def safe_divide(numerator, denominator, fill_value=np.nan):
    denominator = np.where(denominator == 0, np.nan, denominator)
    result = numerator / denominator
    if not pd.isna(fill_value):
        result = np.where(np.isnan(result), fill_value, result)
    return result


def get_internal_task_total(num_chunks):
    return 8 + num_chunks


def get_pipeline_task_total(num_chunks):
    return 10 + num_chunks


def prepare_dataframe(raw_df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = raw_df.copy()
    working = working.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if working.empty:
        raise ValueError("No rows remain after dropping NaNs from required columns.")

    return working


def add_datetime_and_filter(working_df):
    working = working_df.copy()
    working["DateTime"] = pd.to_datetime(working["Timestamp"], unit="s", errors="coerce", utc=True).dt.tz_localize(None)
    working = working.dropna(subset=["DateTime"]).sort_values("Timestamp").reset_index(drop=True)

    if POST_2018_ONLY:
        working = working[working["DateTime"] >= pd.Timestamp(POST_2018_START)].reset_index(drop=True)

    if working.empty:
        raise ValueError("No rows remain after datetime parsing and date filtering.")

    return working


def first_hit_bar(hit_mask):
    hit_any = hit_mask.any(axis=1)
    first_bar = np.where(hit_any, hit_mask.argmax(axis=1) + 1, -1).astype(np.int16)
    return first_bar, hit_any


def build_labeled_dataframe(df, include_diagnostics=INCLUDE_DIAGNOSTIC_COLUMNS, show_progress=False, progress=None):
    owns_progress = progress is None
    if progress is None:
        progress = StageProgress(enabled=show_progress)

    try:
        progress.start("Clean/filter data")
        working = prepare_dataframe(df)
        progress.done()

        progress.start("Compute datetime helpers")
        working = add_datetime_and_filter(working)
        n_rows = len(working)
        candidate_end = max(0, n_rows - HORIZON)
        num_chunks = math.ceil(candidate_end / CHUNK_SIZE) if candidate_end > 0 else 0
        total_tasks = get_internal_task_total(num_chunks) if owns_progress else get_pipeline_task_total(num_chunks)
        progress.set_total(total_tasks)
        progress.done()

        progress.start("Compute ATR/context")
        open_arr = working["Open"].astype(np.float64).to_numpy()
        high_arr = working["High"].astype(np.float64).to_numpy()
        low_arr = working["Low"].astype(np.float64).to_numpy()
        close_arr = working["Close"].astype(np.float64).to_numpy()

        close_series = pd.Series(close_arr)
        true_range = compute_true_range(
            pd.Series(high_arr),
            pd.Series(low_arr),
            close_series,
        )
        atr_arr = wilder_smoothing(true_range, ATR_PERIOD).to_numpy(dtype=np.float64)
        ema_arr = close_series.ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().to_numpy(dtype=np.float64)
        ret_short_arr = close_series.pct_change(CONTEXT_RET_SHORT).to_numpy(dtype=np.float64)
        ret_long_arr = close_series.pct_change(CONTEXT_RET_LONG).to_numpy(dtype=np.float64)
        trend_score_arr = np.sign(ret_short_arr) + np.sign(ret_long_arr) + np.sign(close_arr - ema_arr)
        context_up_arr = trend_score_arr >= 2
        context_down_arr = trend_score_arr <= -2
        progress.done()

        progress.start("Initialize output arrays")
        min_valid_index = max(ATR_PERIOD, TREND_EMA_PERIOD, CONTEXT_RET_SHORT, CONTEXT_RET_LONG)
        valid_mask = np.zeros(n_rows, dtype=bool)
        if candidate_end > min_valid_index:
            valid_mask[min_valid_index:candidate_end] = True

        valid_mask &= np.isfinite(atr_arr)
        valid_mask &= np.isfinite(ema_arr)
        valid_mask &= np.isfinite(ret_short_arr)
        valid_mask &= np.isfinite(ret_long_arr)

        label_available = np.zeros(n_rows, dtype=np.int8)
        state_class = np.full(n_rows, -1, dtype=np.int8)

        entry_price = np.full(n_rows, np.nan, dtype=np.float64)
        atr_at_label = np.full(n_rows, np.nan, dtype=np.float64)
        trend_score_diag = np.full(n_rows, np.nan, dtype=np.float32)
        context_state = np.full(n_rows, "unlabeled", dtype=object)
        up_target_hit_bar = np.full(n_rows, -1, dtype=np.int16)
        up_stop_hit_bar = np.full(n_rows, -1, dtype=np.int16)
        down_target_hit_bar = np.full(n_rows, -1, dtype=np.int16)
        down_stop_hit_bar = np.full(n_rows, -1, dtype=np.int16)
        horizon_return_atr = np.full(n_rows, np.nan, dtype=np.float64)
        favorable_fraction_up = np.full(n_rows, np.nan, dtype=np.float32)
        favorable_fraction_down = np.full(n_rows, np.nan, dtype=np.float32)
        first_touch_side = np.full(n_rows, "unlabeled", dtype=object)
        up_opportunity_diag = np.zeros(n_rows, dtype=np.int8)
        down_opportunity_diag = np.zeros(n_rows, dtype=np.int8)
        progress.done()

        if candidate_end > 0:
            future_high_view = sliding_window_view(high_arr[1:], HORIZON)
            future_low_view = sliding_window_view(low_arr[1:], HORIZON)
            future_close_view = sliding_window_view(close_arr[1:], HORIZON)
            relative_positions = np.arange(1, HORIZON + 1, dtype=np.int16)[None, :]
        else:
            future_high_view = None
            future_low_view = None
            future_close_view = None
            relative_positions = None

        for chunk_index, start in enumerate(range(0, candidate_end, CHUNK_SIZE), start=1):
            end = min(start + CHUNK_SIZE, candidate_end)
            progress.start(f"Label chunk {chunk_index}/{num_chunks}")

            row_idx = np.arange(start, end, dtype=np.int64)
            chunk_valid = valid_mask[start:end]

            if chunk_valid.any():
                future_high = future_high_view[start:end]
                future_low = future_low_view[start:end]
                future_close = future_close_view[start:end]

                entry_chunk = open_arr[start + 1:end + 1]
                atr_chunk = atr_arr[start:end]
                up_target = entry_chunk + TARGET_MULT * atr_chunk
                down_target = entry_chunk - TARGET_MULT * atr_chunk
                up_stop = entry_chunk - STOP_MULT * atr_chunk
                down_stop = entry_chunk + STOP_MULT * atr_chunk

                up_target_mask = future_high >= up_target[:, None]
                up_stop_mask = future_low <= up_stop[:, None]
                down_target_mask = future_low <= down_target[:, None]
                down_stop_mask = future_high >= down_stop[:, None]

                up_target_first, up_target_any = first_hit_bar(up_target_mask)
                up_stop_first, up_stop_any = first_hit_bar(up_stop_mask)
                down_target_first, down_target_any = first_hit_bar(down_target_mask)
                down_stop_first, down_stop_any = first_hit_bar(down_stop_mask)

                up_first_touch_ok = up_target_any & ((~up_stop_any) | (up_target_first < up_stop_first))
                down_first_touch_ok = down_target_any & ((~down_stop_any) | (down_target_first < down_stop_first))

                horizon_return_chunk = safe_divide(future_close[:, -1] - entry_chunk, atr_chunk)
                favorable_up_chunk = (future_close > entry_chunk[:, None]).mean(axis=1)
                favorable_down_chunk = (future_close < entry_chunk[:, None]).mean(axis=1)

                up_immediate_invalidation = np.where(
                    up_target_first > 0,
                    (
                        (relative_positions > up_target_first[:, None])
                        & (relative_positions <= (up_target_first[:, None] + AMBIGUOUS_REVERSAL_WINDOW))
                        & (future_low <= up_stop[:, None])
                    ).any(axis=1),
                    False,
                )
                down_immediate_invalidation = np.where(
                    down_target_first > 0,
                    (
                        (relative_positions > down_target_first[:, None])
                        & (relative_positions <= (down_target_first[:, None] + AMBIGUOUS_REVERSAL_WINDOW))
                        & (future_high >= down_stop[:, None])
                    ).any(axis=1),
                    False,
                )

                up_opportunity_chunk = (
                    chunk_valid
                    & up_first_touch_ok
                    & (horizon_return_chunk >= MIN_HORIZON_RETURN_MULT)
                    & (favorable_up_chunk >= MIN_FAVORABLE_OCCUPANCY)
                    & (~up_immediate_invalidation)
                )
                down_opportunity_chunk = (
                    chunk_valid
                    & down_first_touch_ok
                    & (horizon_return_chunk <= -MIN_HORIZON_RETURN_MULT)
                    & (favorable_down_chunk >= MIN_FAVORABLE_OCCUPANCY)
                    & (~down_immediate_invalidation)
                )

                chunk_state = np.full(end - start, -1, dtype=np.int8)
                chunk_state[chunk_valid] = 0

                exclusive_up = up_opportunity_chunk & (~down_opportunity_chunk)
                exclusive_down = down_opportunity_chunk & (~up_opportunity_chunk)
                chunk_state[exclusive_up & context_down_arr[start:end]] = 3
                chunk_state[exclusive_up & (~context_down_arr[start:end])] = 1
                chunk_state[exclusive_down & context_up_arr[start:end]] = 4
                chunk_state[exclusive_down & (~context_up_arr[start:end])] = 2

                label_available[row_idx[chunk_valid]] = 1
                state_class[row_idx[chunk_valid]] = chunk_state[chunk_valid]

                target_first_up = np.where(up_target_any, up_target_first, 32767)
                target_first_down = np.where(down_target_any, down_target_first, 32767)
                first_touch_chunk = np.full(end - start, "none", dtype=object)
                first_touch_chunk[(target_first_up < target_first_down)] = "up"
                first_touch_chunk[(target_first_down < target_first_up)] = "down"
                first_touch_chunk[(target_first_up == target_first_down) & up_target_any & down_target_any] = "both"

                context_chunk = np.full(end - start, "neutral", dtype=object)
                context_chunk[context_up_arr[start:end]] = "up"
                context_chunk[context_down_arr[start:end]] = "down"

                entry_price[row_idx[chunk_valid]] = entry_chunk[chunk_valid]
                atr_at_label[row_idx[chunk_valid]] = atr_chunk[chunk_valid]
                trend_score_diag[row_idx[chunk_valid]] = trend_score_arr[start:end][chunk_valid].astype(np.float32)
                context_state[row_idx[chunk_valid]] = context_chunk[chunk_valid]
                up_target_hit_bar[row_idx[chunk_valid]] = up_target_first[chunk_valid]
                up_stop_hit_bar[row_idx[chunk_valid]] = up_stop_first[chunk_valid]
                down_target_hit_bar[row_idx[chunk_valid]] = down_target_first[chunk_valid]
                down_stop_hit_bar[row_idx[chunk_valid]] = down_stop_first[chunk_valid]
                horizon_return_atr[row_idx[chunk_valid]] = horizon_return_chunk[chunk_valid]
                favorable_fraction_up[row_idx[chunk_valid]] = favorable_up_chunk[chunk_valid].astype(np.float32)
                favorable_fraction_down[row_idx[chunk_valid]] = favorable_down_chunk[chunk_valid].astype(np.float32)
                first_touch_side[row_idx[chunk_valid]] = first_touch_chunk[chunk_valid]
                up_opportunity_diag[row_idx[chunk_valid]] = up_opportunity_chunk[chunk_valid].astype(np.int8)
                down_opportunity_diag[row_idx[chunk_valid]] = down_opportunity_chunk[chunk_valid].astype(np.int8)

            progress.done()

        progress.start("Assemble label columns")
        state_label = np.array([CLASS_ID_TO_NAME[class_id] for class_id in state_class], dtype=object)
        no_trade_label = (state_class == 0).astype(np.int8)
        trend_up_label = (state_class == 1).astype(np.int8)
        trend_down_label = (state_class == 2).astype(np.int8)
        reversal_up_label = (state_class == 3).astype(np.int8)
        reversal_down_label = (state_class == 4).astype(np.int8)

        output_df = working.copy()
        output_df["label_available"] = label_available
        output_df["state_class"] = state_class
        output_df["state_label"] = state_label
        output_df["no_trade_label"] = no_trade_label
        output_df["trend_up_label"] = trend_up_label
        output_df["trend_down_label"] = trend_down_label
        output_df["reversal_up_label"] = reversal_up_label
        output_df["reversal_down_label"] = reversal_down_label
        progress.done()

        progress.start("Attach diagnostics")
        if include_diagnostics:
            output_df["entry_price"] = pd.to_numeric(entry_price, downcast="float")
            output_df["atr_at_label"] = pd.to_numeric(atr_at_label, downcast="float")
            output_df["trend_score"] = pd.to_numeric(trend_score_diag, downcast="float")
            output_df["context_state"] = context_state
            output_df["up_target_hit_bar"] = up_target_hit_bar
            output_df["up_stop_hit_bar"] = up_stop_hit_bar
            output_df["down_target_hit_bar"] = down_target_hit_bar
            output_df["down_stop_hit_bar"] = down_stop_hit_bar
            output_df["horizon_return_atr"] = pd.to_numeric(horizon_return_atr, downcast="float")
            output_df["favorable_fraction_up"] = pd.to_numeric(favorable_fraction_up, downcast="float")
            output_df["favorable_fraction_down"] = pd.to_numeric(favorable_fraction_down, downcast="float")
            output_df["first_touch_side"] = first_touch_side
            output_df["up_opportunity"] = up_opportunity_diag
            output_df["down_opportunity"] = down_opportunity_diag
        progress.done()

        progress.start("Validate labels")
        one_hot_sum = output_df[CLASS_OUTPUT_COLUMNS].sum(axis=1)
        valid_rows = output_df["label_available"] == 1
        invalid_rows = ~valid_rows

        if not (one_hot_sum[valid_rows] == 1).all():
            raise ValueError("Found valid rows without exactly one active class label.")
        if not (one_hot_sum[invalid_rows] == 0).all():
            raise ValueError("Found invalid rows with active class labels.")
        if not (output_df.loc[invalid_rows, "state_class"] == -1).all():
            raise ValueError("Found invalid rows with labeled state_class.")
        if not (output_df.loc[invalid_rows, "state_label"] == "unlabeled").all():
            raise ValueError("Found invalid rows with labeled state_label.")
        progress.done()

        progress.start("Distribution diagnostics")
        valid_count = int(valid_rows.sum())
        total_count = len(output_df)

        print("\n===== LABEL V2 SUMMARY =====")
        print(f"Rows: {total_count:,}")
        print(f"Valid label rows: {valid_count:,} ({(valid_count / total_count):.2%})")
        print(f"Unlabeled edge rows: {total_count - valid_count:,}")
        print(f"Config: HORIZON={HORIZON}, ATR_PERIOD={ATR_PERIOD}, TARGET_MULT={TARGET_MULT}, STOP_MULT={STOP_MULT}")

        print("\n===== CLASS DISTRIBUTION (VALID ROWS ONLY) =====")
        if valid_count == 0:
            print("No valid rows available for labeling.")
        else:
            counts = {
                "no_trade": int(output_df.loc[valid_rows, "no_trade_label"].sum()),
                "trend_up": int(output_df.loc[valid_rows, "trend_up_label"].sum()),
                "trend_down": int(output_df.loc[valid_rows, "trend_down_label"].sum()),
                "reversal_up": int(output_df.loc[valid_rows, "reversal_up_label"].sum()),
                "reversal_down": int(output_df.loc[valid_rows, "reversal_down_label"].sum()),
            }
            for class_name, class_count in counts.items():
                print(f"{class_name:16s}: {class_count:>10,} ({(class_count / valid_count):.2%})")

            small_classes = [name for name, count in counts.items() if count > 0 and (count / valid_count) < 0.01]
            empty_classes = [name for name, count in counts.items() if count == 0]

            if empty_classes:
                print(f"\nWARNING: Empty classes detected: {', '.join(empty_classes)}")
            if small_classes:
                print(f"WARNING: Very small classes (<1%): {', '.join(small_classes)}")
            if (counts["no_trade"] / valid_count) > 0.80:
                print("WARNING: no_trade dominates more than 80% of valid rows.")
        progress.done()

        return output_df
    finally:
        if owns_progress:
            progress.close()


def run_pipeline(
    data_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    include_diagnostics=INCLUDE_DIAGNOSTIC_COLUMNS,
    show_progress=SHOW_PROGRESS,
):
    progress = StageProgress(enabled=show_progress)

    try:
        progress.start("Load data")
        df = pd.read_csv(data_path)
        progress.done()

        result_df = build_labeled_dataframe(
            df=df,
            include_diagnostics=include_diagnostics,
            show_progress=False,
            progress=progress,
        )

        progress.start("Write CSV")
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved labeled dataset to: {output_path}")
        progress.done()

        return result_df
    finally:
        progress.close()


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
