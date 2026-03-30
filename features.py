import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

INPUT_PATH = "labeled_data.csv"
OUTPUT_PATH = "features.csv"

# Feature windows (in periods/minutes)
SHORT_WINDOW = 5
MID_WINDOW = 10
LONG_WINDOW = 20
VOLATILITY_WINDOW = 10
VOLUME_WINDOW = 20

# ================================
# LOAD DATA
# ================================

df = pd.read_csv(INPUT_PATH)
df = df.dropna().reset_index(drop=True)

print(f"Loaded {len(df)} samples")

# Extract OHLCV
open_prices = df['Open'].values
high_prices = df['High'].values
low_prices = df['Low'].values
close_prices = df['Close'].values
volume = df['Volume'].values

n = len(df)

# ================================
# INITIALIZE FEATURE ARRAYS
# ================================

features = {
    # Returns (momentum)
    'return_1m': np.full(n, np.nan),           # 1-minute returns
    'return_sma_5': np.full(n, np.nan),        # avg return over 5 periods
    'return_sma_10': np.full(n, np.nan),       # avg return over 10 periods
    
    # High-Low range (volatility / price action)
    'hl_range': np.full(n, np.nan),            # (high - low) / close
    'hl_range_sma': np.full(n, np.nan),        # rolling avg of hl_range
    
    # Close position in range (momentum)
    'close_position': np.full(n, np.nan),      # (close - low) / (high - low)
    
    # Open-Close comparison
    'oc_ratio': np.full(n, np.nan),            # (close - open) / open
    
    # Moving averages
    'ma_5': np.full(n, np.nan),
    'ma_10': np.full(n, np.nan),
    'ma_20': np.full(n, np.nan),
    
    # Price relative to MAs (trend alignment)
    'price_above_ma5': np.full(n, np.nan),     # (close - ma5) / ma5
    'price_above_ma10': np.full(n, np.nan),    # (close - ma10) / ma10
    
    # Rate of change
    'roc_5': np.full(n, np.nan),               # (close[i] - close[i-5]) / close[i-5]
    'roc_10': np.full(n, np.nan),
    
    # Volatility (std of returns)
    'volatility_5': np.full(n, np.nan),
    'volatility_10': np.full(n, np.nan),
    
    # Volume momentum
    'volume_sma': np.full(n, np.nan),
    'volume_ratio': np.full(n, np.nan),        # current vol / avg vol
    
    # HSL (High-Low) smoothed range
    'hsl_5': np.full(n, np.nan),               # rolling max of hl_range
    
    # Technical Indicators
    # RSI (Relative Strength Index)
    'rsi_5': np.full(n, np.nan),
    'rsi_10': np.full(n, np.nan),
    'rsi_14': np.full(n, np.nan),
    
    # MACD (Moving Average Convergence Divergence)
    'macd_12_26': np.full(n, np.nan),          # 12-period EMA - 26-period EMA
    'macd_signal_9': np.full(n, np.nan),       # 9-period EMA of MACD
    'macd_histogram': np.full(n, np.nan),      # MACD - Signal
    
    # Bollinger Bands
    'bb_upper_20': np.full(n, np.nan),         # 20-period MA + 2*std
    'bb_lower_20': np.full(n, np.nan),         # 20-period MA - 2*std
    'bb_position': np.full(n, np.nan),         # (close - lower) / (upper - lower)
    'bb_width': np.full(n, np.nan),            # (upper - lower) / middle
    
    # ATR (Average True Range) - volatility measure
    'atr_14': np.full(n, np.nan),
    
    # Stochastic Oscillator
    'stoch_k_14': np.full(n, np.nan),          # %K
    'stoch_d_3': np.full(n, np.nan),           # %D (3-period SMA of %K)
    
    # CCI (Commodity Channel Index) - trend/cyclical moves
    'cci_20': np.full(n, np.nan),
    
    # ADX (Average Directional Index) - trend strength (simplified)
    'di_plus_14': np.full(n, np.nan),          # Positive Directional Indicator
    'di_minus_14': np.full(n, np.nan),         # Negative Directional Indicator
}

# ================================
# FEATURE CALCULATION
# ================================

print("Calculating features...")

# Determine minimum period needed for all indicators
MIN_PERIODS = LONG_WINDOW  # We need at least 20 periods for all features to be meaningful

for i in tqdm(range(n)):
    # Skip early rows (not enough data for indicator calculation)
    if i < MIN_PERIODS:
        continue
    # 1-minute return
    if i > 0:
        features['return_1m'][i] = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
    
    # High-Low range (normalized by close)
    features['hl_range'][i] = (high_prices[i] - low_prices[i]) / close_prices[i]
    
    # Close position in range (0=low, 1=high)
    if high_prices[i] != low_prices[i]:
        features['close_position'][i] = (close_prices[i] - low_prices[i]) / (high_prices[i] - low_prices[i])
    else:
        features['close_position'][i] = 0.5
    
    # Open-Close ratio
    features['oc_ratio'][i] = (close_prices[i] - open_prices[i]) / open_prices[i]
    
    # Short window (5 periods)
    if i >= SHORT_WINDOW - 1:
        window_data = close_prices[i-SHORT_WINDOW+1:i+1]
        features['ma_5'][i] = np.mean(window_data)
        features['return_sma_5'][i] = np.mean(np.diff(window_data) / window_data[:-1])
        
        hl_range_window = (high_prices[i-SHORT_WINDOW+1:i+1] - low_prices[i-SHORT_WINDOW+1:i+1]) / close_prices[i-SHORT_WINDOW+1:i+1]
        features['hl_range_sma'][i] = np.mean(hl_range_window)
        
        if features['ma_5'][i] > 0:
            features['price_above_ma5'][i] = (close_prices[i] - features['ma_5'][i]) / features['ma_5'][i]
        
        features['roc_5'][i] = (close_prices[i] - close_prices[i-SHORT_WINDOW]) / close_prices[i-SHORT_WINDOW]
        
        ret_5 = (close_prices[i-SHORT_WINDOW+1:i+1] - close_prices[i-SHORT_WINDOW:i]) / close_prices[i-SHORT_WINDOW:i]
        features['volatility_5'][i] = np.std(ret_5)
        
        vol_window = volume[i-SHORT_WINDOW+1:i+1]
        features['volume_sma'][i] = np.mean(vol_window)
        if features['volume_sma'][i] > 0:
            features['volume_ratio'][i] = volume[i] / features['volume_sma'][i]
    
    # Mid window (10 periods)
    if i >= MID_WINDOW - 1:
        window_data = close_prices[i-MID_WINDOW+1:i+1]
        features['ma_10'][i] = np.mean(window_data)
        features['return_sma_10'][i] = np.mean(np.diff(window_data) / window_data[:-1])
        
        if features['ma_10'][i] > 0:
            features['price_above_ma10'][i] = (close_prices[i] - features['ma_10'][i]) / features['ma_10'][i]
        
        features['roc_10'][i] = (close_prices[i] - close_prices[i-MID_WINDOW]) / close_prices[i-MID_WINDOW]
        
        ret_10 = (close_prices[i-MID_WINDOW+1:i+1] - close_prices[i-MID_WINDOW:i]) / close_prices[i-MID_WINDOW:i]
        features['volatility_10'][i] = np.std(ret_10)
    
    # Long window (20 periods)
    if i >= LONG_WINDOW - 1:
        window_data = close_prices[i-LONG_WINDOW+1:i+1]
        features['ma_20'][i] = np.mean(window_data)
    
    # High-Low ratio max (smoothed resistance/support reference)
    if i >= SHORT_WINDOW - 1:
        hl_window = (high_prices[i-SHORT_WINDOW+1:i+1] - low_prices[i-SHORT_WINDOW+1:i+1]) / close_prices[i-SHORT_WINDOW+1:i+1]
        features['hsl_5'][i] = np.max(hl_window)

# ================================
# TECHNICAL INDICATORS CALCULATION
# ================================

print("Calculating technical indicators...")

for i in tqdm(range(n)):
    # === RSI (Relative Strength Index) ===
    for rsi_period in [5, 10, 14]:
        if i >= rsi_period:
            window_close = close_prices[i-rsi_period+1:i+1]
            deltas = np.diff(window_close)
            seed = deltas[:rsi_period].mean()
            
            up = deltas.copy()
            up[up < 0] = 0
            down = -deltas.copy()
            down[down < 0] = 0
            
            rs = np.mean(up) / np.mean(down) if np.mean(down) != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
            
            features[f'rsi_{rsi_period}'][i] = rsi
    
    # === MACD (simplified: no EMA smoothing, just SMA) ===
    if i >= 25:  # Needs at least 26 periods
        ema_12 = np.mean(close_prices[i-11:i+1])
        ema_26 = np.mean(close_prices[i-25:i+1])
        features['macd_12_26'][i] = ema_12 - ema_26
        
        # Signal line (9-period average of MACD)
        if i >= 34:  # Needs 26 + 9 periods
            macd_window = []
            for j in range(i-8, i+1):
                if j >= 25:
                    e12 = np.mean(close_prices[j-11:j+1])
                    e26 = np.mean(close_prices[j-25:j+1])
                    macd_window.append(e12 - e26)
            
            if len(macd_window) == 9:
                features['macd_signal_9'][i] = np.mean(macd_window)
                features['macd_histogram'][i] = features['macd_12_26'][i] - features['macd_signal_9'][i]
    
    # === Bollinger Bands (20-period) ===
    if i >= 19:
        window_close = close_prices[i-19:i+1]
        sma_20 = np.mean(window_close)
        std_20 = np.std(window_close)
        
        features['bb_upper_20'][i] = sma_20 + 2 * std_20
        features['bb_lower_20'][i] = sma_20 - 2 * std_20
        
        bb_range = features['bb_upper_20'][i] - features['bb_lower_20'][i]
        if bb_range > 0:
            features['bb_position'][i] = (close_prices[i] - features['bb_lower_20'][i]) / bb_range
            features['bb_width'][i] = bb_range / sma_20
    
    # === ATR (Average True Range - 14 period) ===
    if i >= 13:
        tr_window = []
        for j in range(i-13, i+1):
            if j == 0:
                tr = high_prices[j] - low_prices[j]
            else:
                tr = max(
                    high_prices[j] - low_prices[j],
                    abs(high_prices[j] - close_prices[j-1]),
                    abs(low_prices[j] - close_prices[j-1])
                )
            tr_window.append(tr)
        
        features['atr_14'][i] = np.mean(tr_window)
    
    # === Stochastic Oscillator (14-period K, 3-period D) ===
    if i >= 13:
        window_high = high_prices[i-13:i+1]
        window_low = low_prices[i-13:i+1]
        
        highest = np.max(window_high)
        lowest = np.min(window_low)
        
        if highest != lowest:
            stoch_k = 100 * (close_prices[i] - lowest) / (highest - lowest)
        else:
            stoch_k = 50
        
        features['stoch_k_14'][i] = stoch_k
        
        # %D is 3-period SMA of %K
        if i >= 15:  # Needs 14 + 3 periods
            k_window = []
            for j in range(i-2, i+1):
                if j >= 13:
                    h = np.max(high_prices[j-13:j+1])
                    l = np.min(low_prices[j-13:j+1])
                    k = 100 * (close_prices[j] - l) / (h - l) if h != l else 50
                    k_window.append(k)
            
            if len(k_window) == 3:
                features['stoch_d_3'][i] = np.mean(k_window)
    
    # === CCI (Commodity Channel Index - 20 period) ===
    if i >= 19:
        tp_window = (high_prices[i-19:i+1] + low_prices[i-19:i+1] + close_prices[i-19:i+1]) / 3
        tp = (high_prices[i] + low_prices[i] + close_prices[i]) / 3
        sma_tp = np.mean(tp_window)
        mad_tp = np.mean(np.abs(tp_window - sma_tp))
        
        if mad_tp != 0:
            features['cci_20'][i] = (tp - sma_tp) / (0.015 * mad_tp)
        else:
            features['cci_20'][i] = 0
    
    # === Directional Indicators (ADX simplified - 14 period) ===
    if i >= 13:
        di_plus_window = 0
        di_minus_window = 0
        
        for j in range(i-13, i+1):
            if j == 0:
                up_move = high_prices[j]
                down_move = low_prices[j]
            else:
                up_move = high_prices[j] - high_prices[j-1]
                down_move = low_prices[j-1] - low_prices[j]
            
            if up_move > 0 and up_move > down_move:
                di_plus_window += up_move
            if down_move > 0 and down_move > up_move:
                di_minus_window += down_move
        
        atr_val = features['atr_14'][i]
        if atr_val > 0:
            features['di_plus_14'][i] = 100 * (di_plus_window / 14) / atr_val
            features['di_minus_14'][i] = 100 * (di_minus_window / 14) / atr_val
        else:
            features['di_plus_14'][i] = 0
            features['di_minus_14'][i] = 0

# ================================
# CREATE FEATURE DATAFRAME
# ================================

features_df = pd.DataFrame(features)

# Combine with original data
result_df = pd.concat([df, features_df], axis=1)

# ================================
# FEATURE STATISTICS & DIAGNOSTICS
# ================================

print("\n===== FEATURE STATISTICS =====")
print(features_df.describe())

print("\n===== NAN COUNT =====")
nan_counts = features_df.isna().sum()
print(nan_counts[nan_counts > 0])  # Only show features with NaNs

print("\n===== FEATURE CORRELATION WITH LABELS =====")
for label in ['buy_label', 'sell_label', 'idk_label']:
    print(f"\n{label}:")
    correlations = features_df.corrwith(df[label]).sort_values(ascending=False)
    print(correlations.head(10))

# ================================
# SAVE OUTPUT
# ================================

result_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✓ Features saved to: {OUTPUT_PATH}")
print(f"Total columns: {len(result_df.columns)}")
print(f"Feature count: {len(features_df.columns)}")
print(f"Total samples: {len(result_df)}")
print(f"Samples with NaN features (warmup period): {features_df.isna().any(axis=1).sum()}")
print(f"Usable samples for training: {(~features_df.isna().any(axis=1)).sum()}")

# ================================
# MEMORY CHECK
# ================================

mem_usage = result_df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage: {mem_usage:.2f} MB")
