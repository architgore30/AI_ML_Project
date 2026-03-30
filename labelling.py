import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "dataset.csv"
OUTPUT_PATH = "labeled_data.csv"

TP = 0.015       # +1.5% → strong upward trend (BUY)
SL = 0.007       # -0.7% → early downside detection (SELL)
MAX_HORIZON = 30

# ================================
# LOAD DATA
# ================================

df = pd.read_csv(DATA_PATH)
df = df.dropna().reset_index(drop=True)

# testing on smaller subset first
df = df.tail(200_000)

close = df['close'].values
n = len(df)

buy_labels = np.zeros(n, dtype=np.int8)
sell_labels = np.zeros(n, dtype=np.int8)

# ================================
# LABEL GENERATION
# ================================

for i in tqdm(range(n - MAX_HORIZON), desc="Generating labels"):
    current_price = close[i]

    upper_threshold = current_price * (1 + TP)
    lower_threshold = current_price * (1 - SL)

    future_prices = close[i+1:i+1+MAX_HORIZON]

    for price in future_prices:
        if price >= upper_threshold:
            buy_labels[i] = 1
            break

        elif price <= lower_threshold:
            sell_labels[i] = 1
            break

# ================================
# ATTACH LABELS
# ================================

df['buy_label'] = buy_labels
df['sell_label'] = sell_labels

# ================================
# DIAGNOSTICS
# ================================

total = len(df)

buy_ratio = df['buy_label'].sum() / total
sell_ratio = df['sell_label'].sum() / total
no_trade_ratio = 1 - (buy_ratio + sell_ratio)

print("\n===== LABEL DISTRIBUTION =====")
print(f"BUY: {buy_ratio:.4f}")
print(f"SELL: {sell_ratio:.4f}")
print(f"NO TRADE: {no_trade_ratio:.4f}")

overlap = ((df['buy_label'] == 1) & (df['sell_label'] == 1)).sum()
print(f"Overlap (should be 0): {overlap}")

# ================================
# SAVE OUTPUT
# ================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nLabeled dataset saved to: {OUTPUT_PATH}")