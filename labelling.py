"""
Triple-Barrier Event Labeling for Bitcoin Trend Detection

Purpose:
    Generate ground-truth trading signals using triple-barrier labeling:
    - BUY signal: Price rises +TP% within MAX_HORIZON window (confidence in uptrend)
    - SELL signal: Price drops -SL% within MAX_HORIZON window (early risk warning)
                   OR price stays below entry for 70%+ of horizon (sustained downtrend)
    - IDK signal: Neither threshold hit (market too choppy to trade)

Philosophy:
    Risk-first labeling with asymmetric thresholds (TP > SL magnitude).
    This creates intentional class imbalance:
    - Harder to trigger BUY (requires strong conviction) → fewer, higher-quality signals
    - Easier to trigger SELL (detects weakness early AND sustained downtrends)
      → more signals for risk management and position exit

Output:
    labeled_data.csv with three mutually-exclusive binary columns:
    - buy_label, sell_label, idk_label (exactly one = 1 per row)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "dataset.csv"
OUTPUT_PATH = "labeled_data.csv"

# ===== Triple-Barrier Thresholds =====
# Asymmetric by design - controls signal sensitivity:
# - TP: Harder threshold for BUY (requires conviction)
# - SL: Easier threshold for SELL (protects against downside)
# Empirically chosen based on 30-min window price movement statistics:
# - Median high: +0.1809%, Median low: -0.1844%
# - TP at 0.18% sits at ~50th percentile of upward moves
# - SL at 0.15% sits slightly below median downward move magnitude
TP = 0.0018        # 0.18% upward move → BUY signal
SL = 0.0015        # 0.15% downward move → SELL signal (reversal)
MAX_HORIZON = 30   # minutes to detect threshold hit
DOWNTREND_THRESHOLD = 0.70  # fraction of horizon bars that must be below entry for sustained downtrend sell

# ================================
# LOAD DATA
# ================================

# Load raw Bitcoin data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {len(df)} raw samples")

df = df.dropna().reset_index(drop=True)
print(f"   Cleaned NaN values: {len(df)} samples remain")

# ====== REGIME FILTER: Post-2018 Data Only ======
# Focus on modern cryptocurrency market behavior.
# Pre-2018 data has structural differences (low volume, different market structure).
# Timestamp is Unix seconds in this dataset.
df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df[df['DateTime'] >= '2018-01-01']
print(f"   Regime filter (post-2018): {len(df)} samples remain")

# Optional: Uncomment for faster debug iterations during development
# df = df.tail(200000)  # Tests on 200k samples instead of full dataset

close = df['Close'].values
n = len(df)

print(f"\n⏷ Initializing label arrays...")
buy_labels = np.zeros(n, dtype=np.int8)
sell_labels = np.zeros(n, dtype=np.int8)
# idk_label: "I don't know" - model outputs this when market is uncertain/choppy
idk_labels = np.zeros(n, dtype=np.int8)

# ================================
# LABEL GENERATION (TRIPLE-BARRIER)
# ================================
# For each timestamp i, look ahead up to MAX_HORIZON minutes:
# - If price hits upper_threshold first → buy_label = 1 (strong uptrend detected)
# - Else if price hits lower_threshold first → sell_label = 1 (sharp reversal detected)
# - Else if price stays below entry for 70%+ of horizon → sell_label = 1 (sustained downtrend)
# - Else neither threshold → idk_label = 1 (market too noisy, no clear signal)

i = 0
pbar = tqdm(total=n - MAX_HORIZON, desc="Generating labels")

sell_reversal_count = 0
sell_downtrend_count = 0

while i < n - MAX_HORIZON:
    current_price = close[i]

    # Define threshold prices for this bar
    upper_threshold = current_price * (1 + TP)      # +TP% upward → BUY signal
    lower_threshold = current_price * (1 - SL)      # -SL% downward → SELL signal (reversal)

    # Look ahead within the MAX_HORIZON window
    future_prices = close[i+1:i+1+MAX_HORIZON]

    label = 'idk'
    hit_at = MAX_HORIZON  # default: advance by full horizon if no barrier hit

    for j, price in enumerate(future_prices):
        if price >= upper_threshold:
            label = 'buy'
            hit_at = j + 1
            break
        elif price <= lower_threshold:
            label = 'sell_reversal'
            hit_at = j + 1
            break

    # If no barrier was hit, check for sustained downtrend
    if label == 'idk':
        bars_below_entry = (future_prices < current_price).sum()
        if bars_below_entry >= MAX_HORIZON * DOWNTREND_THRESHOLD:
            label = 'sell_downtrend'

    if label == 'buy':
        buy_labels[i] = 1
    elif label == 'sell_reversal':
        sell_labels[i] = 1
        sell_reversal_count += 1
    elif label == 'sell_downtrend':
        sell_labels[i] = 1
        sell_downtrend_count += 1
    else:
        idk_labels[i] = 1

    # Advance past the event window so overlapping mid-move bars are not re-labelled
    pbar.update(hit_at)
    i += hit_at

pbar.close()

# ================================
# ATTACH LABELS TO DATAFRAME
# ================================

df['buy_label'] = buy_labels
df['sell_label'] = sell_labels
df['idk_label'] = idk_labels

print(f"Labels attached to dataframe")

# ================================
# LABEL DISTRIBUTION ANALYSIS
# ================================
# Verify labels are mutually exclusive and class balance is reasonable

total = len(df)
buy_count = df['buy_label'].sum()
sell_count = df['sell_label'].sum()
idk_count = df['idk_label'].sum()

buy_ratio = buy_count / total
sell_ratio = sell_count / total
idk_ratio = idk_count / total

print(f"\n===== LABEL DISTRIBUTION =====")
print(f"  BUY signals:     {buy_count:,} ({buy_ratio:.2%})")
print(f"  SELL signals:    {sell_count:,} ({sell_ratio:.2%})")
print(f"    of which reversal:  {sell_reversal_count:,} ({sell_reversal_count/total:.2%})")
print(f"    of which downtrend: {sell_downtrend_count:,} ({sell_downtrend_count/total:.2%})")
print(f"  IDK (uncertain): {idk_count:,} ({idk_ratio:.2%})")
print(f"  ---")
print(f"  Total bars:      {total:,}")

# Validate mutual exclusion (critical property)
print(f"Validation checks:")
overlap_buy_sell = ((df['buy_label'] == 1) & (df['sell_label'] == 1)).sum()
overlap_buy_idk = ((df['idk_label'] == 1) & (df['buy_label'] == 1)).sum()
overlap_sell_idk = ((df['idk_label'] == 1) & (df['sell_label'] == 1)).sum()

if overlap_buy_sell == 0 and overlap_buy_idk == 0 and overlap_sell_idk == 0:
    print(f"   Labels are mutually exclusive (no overlap)")
else:
    print(f"   ERROR: Found overlapping labels!")
    print(f"      BUY & SELL overlap:  {overlap_buy_sell}")
    print(f"      BUY & IDK overlap:   {overlap_buy_idk}")
    print(f"      SELL & IDK overlap:  {overlap_sell_idk}")

# ================================
# SAVE OUTPUT
# ================================

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nLabeled dataset saved to: {OUTPUT_PATH}")
print(f"   Columns: buy_label, sell_label, idk_label (+ original feature columns)")
print(f"   Rows: {len(df):,} samples (post-2018 regime)")