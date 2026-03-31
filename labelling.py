"""
Triple-Barrier Event Labeling for Bitcoin Trend Detection

Purpose:
    Generate ground-truth trading signals using triple-barrier labeling:
    - BUY signal: Price rises +TP% within MAX_HORIZON window (confidence in uptrend)
    - SELL signal: Price drops -SL% within MAX_HORIZON window (early risk warning)
    - IDK signal: Neither threshold hit (market too choppy to trade)

Philosophy:
    Risk-first labeling with asymmetric thresholds (TP > SL magnitude).
    This creates intentional class imbalance:
    - Harder to trigger BUY (requires strong conviction) → fewer, higher-quality signals
    - Easier to trigger SELL (detects weakness early) → more signals for risk management

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

# ===== Triple-Barrier Thresholds (Tuned v2) =====
# Asymmetric by design - controls signal sensitivity:
# - TP: Harder threshold for BUY (requires conviction)
# - SL: Easier threshold for SELL (protects against downside)
TP = 0.008       # +0.8% upward move → BUY signal
SL = 0.005       # -0.5% downward move → SELL signal
MAX_HORIZON = 60 # minutes to detect threshold hit (doubled from 30)

# ================================
# LOAD DATA
# ================================

# Load raw Bitcoin data
print(f"📂 Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded {len(df)} raw samples")

df = df.dropna().reset_index(drop=True)
print(f"   ✓ Cleaned NaN values: {len(df)} samples remain")

# ===== REGIME FILTER: Post-2018 Data Only =====
# Focus on modern cryptocurrency market behavior.
# Pre-2018 has structural differences (low volume, different market structure).
# Timestamp is Unix seconds in this dataset.
df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df[df['DateTime'] >= '2018-01-01']
print(f"   ✓ Regime filter (post-2018): {len(df)} samples remain")

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
# - Else if price hits lower_threshold first → sell_label = 1 (downside risk detected)
# - Else neither threshold → idk_label = 1 (market too noisy, no clear signal)

for i in tqdm(range(n - MAX_HORIZON), desc="🏷️  Generating labels"):
    current_price = close[i]

    # Define threshold prices for this bar
    upper_threshold = current_price * (1 + TP)      # +0.8% upward → BUY signal
    lower_threshold = current_price * (1 - SL)      # -0.5% downward → SELL signal

    # Look ahead within the MAX_HORIZON window
    future_prices = close[i+1:i+1+MAX_HORIZON]

    trade_selected = False
    for price in future_prices:
        if price >= upper_threshold:
            buy_labels[i] = 1
            trade_selected = True
            break

        elif price <= lower_threshold:
            sell_labels[i] = 1
            trade_selected = True
            break
    if not trade_selected:
        idk_labels[i] = 1

# ================================
# ATTACH LABELS TO DATAFRAME
# ================================

df['buy_label'] = buy_labels
df['sell_label'] = sell_labels
df['idk_label'] = idk_labels

print(f"✓ Labels attached to dataframe")

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

print(f"\n===== 📊 LABEL DISTRIBUTION =====")
print(f"  BUY signals:     {buy_count:,} ({buy_ratio:.2%})")
print(f"  SELL signals:    {sell_count:,} ({sell_ratio:.2%})")
print(f"  IDK (uncertain): {idk_count:,} ({idk_ratio:.2%})")
print(f"  ---")
print(f"  Total bars:      {total:,}")

# Validate mutual exclusion (critical property)
print(f"\n✓ Validation checks:")
overlap_buy_sell = ((df['buy_label'] == 1) & (df['sell_label'] == 1)).sum()
overlap_buy_idk = ((df['idk_label'] == 1) & (df['buy_label'] == 1)).sum()
overlap_sell_idk = ((df['idk_label'] == 1) & (df['sell_label'] == 1)).sum()

if overlap_buy_sell == 0 and overlap_buy_idk == 0 and overlap_sell_idk == 0:
    print(f"   ✓ Labels are mutually exclusive (no overlap) ✓")
else:
    print(f"   ❌ ERROR: Found overlapping labels!")
    print(f"      BUY & SELL overlap:  {overlap_buy_sell}")
    print(f"      BUY & IDK overlap:   {overlap_buy_idk}")
    print(f"      SELL & IDK overlap:  {overlap_sell_idk}")

# ================================
# SAVE OUTPUT
# ================================

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Labeled dataset saved to: {OUTPUT_PATH}")
print(f"   Columns: buy_label, sell_label, idk_label (+ 35 feature columns)")
print(f"   Rows: {len(df):,} samples (post-2018 regime)")