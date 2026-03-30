import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "dataset.csv"
OUTPUT_PATH = "labeled_data.csv"

TP = 0.008       # +0.8% → more achievable upward target (BUY)
SL = 0.005       # -0.5% → earlier downside detection (SELL)
MAX_HORIZON = 60 # minutes → doubled window for trend formation

# ================================
# LOAD DATA
# ================================

df = pd.read_csv(DATA_PATH)
df = df.dropna().reset_index(drop=True)

# testing on smaller subset first
# df = df.tail(200000)

close = df['Close'].values
n = len(df)

buy_labels = np.zeros(n, dtype=np.int8)
sell_labels = np.zeros(n, dtype=np.int8)
idk_labels = np.zeros(n, dtype=np.int8)     # A third label for the model to jsut say "I don't know" ("idk") when the market is uncertain

# ================================
# LABEL GENERATION
# ================================

for i in tqdm(range(n - MAX_HORIZON), desc="Generating labels"):
    current_price = close[i]

    upper_threshold = current_price * (1 + TP)
    lower_threshold = current_price * (1 - SL)

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
# ATTACH LABELS
# ================================

df['buy_label'] = buy_labels
df['sell_label'] = sell_labels
df['idk_label'] = idk_labels

# ================================
# DIAGNOSTICS
# ================================

total = len(df)

buy_ratio = df['buy_label'].sum() / total
sell_ratio = df['sell_label'].sum() / total
no_trade_ratio = df['idk_label'].sum() / total

print("\n===== LABEL DISTRIBUTION =====")
print(f"BUY: {buy_ratio:.4f}")
print(f"SELL: {sell_ratio:.4f}")
print(f"NO TRADE: {no_trade_ratio:.4f}")

overlap = ((df['buy_label'] == 1) & (df['sell_label'] == 1)).sum()
print(f"Overlap (buy & sell): {overlap}") # should be 0

overlap = ((df['idk_label'] == 1) & (df['buy_label'] == 1)).sum()
print(f"Overlap (idk & buy): {overlap}") # should be 0

overlap = ((df['idk_label'] == 1) & (df['sell_label'] == 1)).sum()
print(f"Overlap (idk & sell): {overlap}") # should be 0

# ================================
# SAVE OUTPUT
# ================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nLabeled dataset saved to: {OUTPUT_PATH}")