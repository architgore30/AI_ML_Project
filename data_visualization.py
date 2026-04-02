"""
Data Visualization & Analysis Script
Purpose: Understand dataset characteristics for debugging and optimization
Current Focus: Volume analysis across entire dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

DATA_PATH = "dataset.csv"
OUTPUT_DIR = "visualizations"
MAX_HORIZON = 120  # Minutes to look ahead for price movement (tunable for testing 20m, 30m, 60m, etc.)

# ================================
# LOAD DATA
# ================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# VOLUME ANALYSIS
# ================================

print("\n===== VOLUME STATISTICS =====")
print(f"Mean volume: {df['Volume'].mean():,.0f}")
print(f"Median volume: {df['Volume'].median():,.0f}")
print(f"Std dev: {df['Volume'].std():,.0f}")
print(f"Min: {df['Volume'].min():,.0f}")
print(f"Max: {df['Volume'].max():,.0f}")

# Calculate volume percentiles to understand distribution
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("\nVolume Percentiles:")
for p in percentiles:
    val = np.percentile(df['Volume'], p)
    print(f"  {p}th percentile: {val:,.0f}")

# ================================
# VOLUME VISUALIZATION
# ================================

print("\nGenerating volume visualizations...")

# Sample every 100k rows for visualization (to prevent memory explosion)
sample_indices = np.arange(len(df))[::100000]
df_sampled = df.iloc[sample_indices].reset_index(drop=True)

print(f"Plotting {len(df_sampled)} sampled points (every 100k rows from {len(df)} total)")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Full volume time series (sampled)
ax1 = axes[0]
ax1.plot(range(len(df_sampled)), df_sampled['Volume'].values, linewidth=1, alpha=0.7, color='blue', marker='o', markersize=3)
ax1.set_title('Bitcoin Volume Over Entire Dataset (Sampled Every 100k rows)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sampled Index (every 100k rows)')
ax1.set_ylabel('Volume')
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')

# Plot 2: Volume distribution (histogram) - use all data for better distribution
ax2 = axes[1]
ax2.hist(df['Volume'].values, bins=100, color='green', alpha=0.7, edgecolor='black')
ax2.set_title('Volume Distribution (Histogram - All Data)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Volume')
ax2.set_ylabel('Frequency')
ax2.set_yscale('log')  # Log scale to see tail
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Volume rolling average (detect trends)
window = 1440  # 1 day of 1-minute data
rolling_vol = df['Volume'].rolling(window=window, center=True).mean()
rolling_vol_sampled = rolling_vol.iloc[sample_indices].reset_index(drop=True)
ax3 = axes[2]
ax3.plot(range(len(df_sampled)), rolling_vol_sampled.values, linewidth=1, alpha=0.8, color='red', marker='o', markersize=2, label=f'{window}-minute MA')
ax3.fill_between(range(len(df_sampled)), rolling_vol_sampled.values, alpha=0.3, color='red')
ax3.set_title(f'Volume Trend (Rolling {window}-minute Average, Sampled Every 100k rows)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Sampled Index (every 100k rows)')
ax3.set_ylabel('Average Volume')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/volume_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/volume_analysis.png")
plt.close()

# ================================
# VOLUME BY TIME PERIODS
# ================================

print("\nAnalyzing volume by periods...")

# Convert timestamp to datetime for period analysis
# The Timestamp column appears to be Unix timestamps
df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Hour'] = df['DateTime'].dt.hour

# Volume by year
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Volume by year
ax1 = axes[0, 0]
yearly_vol = df.groupby('Year')['Volume'].agg(['mean', 'std', 'count'])
ax1.bar(yearly_vol.index, yearly_vol['mean'], color='skyblue', edgecolor='black')
ax1.errorbar(yearly_vol.index, yearly_vol['mean'], yerr=yearly_vol['std'], fmt='none', color='black', capsize=5)
ax1.set_title('Average Volume by Year', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Volume')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Volume by day of week
ax2 = axes[0, 1]
dow_vol = df.groupby('DayOfWeek')['Volume'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax2.bar(range(7), dow_vol.values, color='lightcoral', edgecolor='black')
ax2.set_xticks(range(7))
ax2.set_xticklabels(days)
ax2.set_title('Average Volume by Day of Week', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Volume')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Volume by hour of day
ax3 = axes[1, 0]
hourly_vol = df.groupby('Hour')['Volume'].mean()
if len(hourly_vol) == 24:
    ax3.plot(range(24), hourly_vol.values, marker='o', linewidth=2, markersize=6, color='purple')
    ax3.fill_between(range(24), hourly_vol.values, alpha=0.3, color='purple')
    ax3.set_xticks(range(0, 24, 2))
else:
    # Fallback if hours are missing
    ax3.bar(hourly_vol.index, hourly_vol.values, color='purple', alpha=0.7, edgecolor='black')
ax3.set_title('Average Volume by Hour of Day (UTC)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Hour')
ax3.set_ylabel('Average Volume')
ax3.grid(True, alpha=0.3)

# Plot 4: Data density (samples per year)
ax4 = axes[1, 1]
yearly_count = df.groupby('Year')['Volume'].count()
ax4.bar(yearly_count.index, yearly_count.values, color='lightgreen', edgecolor='black')
ax4.set_title('Number of Samples by Year', fontsize=12, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Sample Count')
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(yearly_count.values):
    ax4.text(yearly_count.index[i], v + 50000, f'{v/1e6:.1f}M', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/volume_by_period.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/volume_by_period.png")
plt.close()

# ================================
# DATA QUALITY INSIGHTS
# ================================

print("\n===== DATA QUALITY ASSESSMENT =====")

# Check for missing or zero volume
zero_vol = (df['Volume'] == 0).sum()
null_vol = df['Volume'].isna().sum()

print(f"Zero volume samples: {zero_vol} ({zero_vol/len(df)*100:.2f}%)")
print(f"Null volume samples: {null_vol} ({null_vol/len(df)*100:.2f}%)")

# Volume volatility
vol_std_normalized = df['Volume'].std() / df['Volume'].mean()
print(f"Volume coefficient of variation: {vol_std_normalized:.2f}")
if vol_std_normalized > 1:
    print("  → High volatility in volume (inconsistent trading activity)")
else:
    print("  → Relatively stable volume")

# Old data analysis
print("\n===== TIME PERIOD ANALYSIS (for old data bias) =====")
pre_2018_vol = df[df['Year'] < 2018]['Volume'].describe()
post_2018_vol = df[df['Year'] >= 2018]['Volume'].describe()

print(f"\nBefore 2018 (older, stable data):")
print(f"  Mean: {pre_2018_vol['mean']:,.0f}")
print(f"  Std: {pre_2018_vol['std']:,.0f}")
print(f"  Samples: {int(pre_2018_vol['count'])}")

print(f"\n2018 onwards (recent, volatile data):")
print(f"  Mean: {post_2018_vol['mean']:,.0f}")
print(f"  Std: {post_2018_vol['std']:,.0f}")
print(f"  Samples: {int(post_2018_vol['count'])}")

ratio = post_2018_vol['mean'] / pre_2018_vol['mean']
print(f"\nRecent volume is {ratio:.1f}x older volume")

# ================================
# PRICE MOVEMENT ANALYSIS (TP/SL TUNING)
# ================================

print(f"\n===== PRICE MOVEMENT ANALYSIS (MAX_HORIZON={MAX_HORIZON} minutes) =====")
print("Calculating highest/lowest prices in each rolling window...")

# Filter to post-2018 data only
df['DateTime'] = pd.to_datetime(df['Timestamp'], unit='s')
df_analysis = df[df['DateTime'] >= '2018-01-01']
print(f"   Regime filter (post-2018): {len(df_analysis)} samples remain")

close = df_analysis['Close'].values
high = df_analysis['High'].values
low = df_analysis['Low'].values
n = len(df_analysis)

# Calculate price multiples for each window
highest_multiples = []
lowest_multiples = []

for i in tqdm(range(n - MAX_HORIZON), desc="Calculating price movements"):
    start_price = close[i]
    
    window_high = np.max(high[i:i+MAX_HORIZON])
    window_low = np.min(low[i:i+MAX_HORIZON])
    
    highest_multiple = window_high / start_price
    lowest_multiple = window_low / start_price
    
    highest_multiples.append(highest_multiple)
    lowest_multiples.append(lowest_multiple)

highest_multiples = np.array(highest_multiples)
lowest_multiples = np.array(lowest_multiples)

# Convert to percentage changes
highest_pct = (highest_multiples - 1) * 100
lowest_pct = (lowest_multiples - 1) * 100

mean = highest_pct.mean()
print(f"\nHighest price in {MAX_HORIZON}-minute window:")
print(f"  Mean: {mean:.4f}% (multiple: {highest_multiples.mean():.6f}x)")
print(f"  Median: {np.median(highest_pct):.4f}%")
print(f"  Std: {highest_pct.std():.4f}%")
print(f"  Rel_Std: {100*highest_pct.std()/mean:.4f}%")
print(f"  Min: {highest_pct.min():.4f}%")
print(f"  Max: {highest_pct.max():.4f}%")
print(f"  Percentiles: 25th={np.percentile(highest_pct, 25):.4f}%, 75th={np.percentile(highest_pct, 75):.4f}%")

lowest_pct.mean()
print(f"\nLowest price in {MAX_HORIZON}-minute window:")
print(f"  Mean: {mean:.4f}% (multiple: {lowest_multiples.mean():.6f}x)")
print(f"  Median: {np.median(lowest_pct):.4f}%")
print(f"  Std: {lowest_pct.std():.4f}%")
print(f"  Rel_Std: {100*lowest_pct.std()/mean:.4f}%")
print(f"  Min: {lowest_pct.min():.4f}%")
print(f"  Max: {lowest_pct.max():.4f}%")
print(f"  Percentiles: 25th={np.percentile(lowest_pct, 25):.4f}%, 75th={np.percentile(lowest_pct, 75):.4f}%")

# Create box plot for TP/SL tuning
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot 1: Highest prices (positive moves, relevant for TP)
ax1 = axes[0]
bp1 = ax1.boxplot([highest_pct], vert=True, patch_artist=True, widths=0.6,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   medianprops=dict(color='darkgreen', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Starting price')
ax1.set_title(f'Highest Price in {MAX_HORIZON}-Minute Window\n(For Take Profit Tuning)', 
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Price Change (%)')
ax1.set_xticklabels(['Upside'])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()

stats_text_up = (f"Mean: {highest_pct.mean():.4f}%\n"
                 f"Median: {np.median(highest_pct):.4f}%\n"
                 f"Q1: {np.percentile(highest_pct, 25):.4f}%\n"
                 f"Q3: {np.percentile(highest_pct, 75):.4f}%\n"
                 f"Std: {highest_pct.std():.4f}%")
ax1.text(1.35, highest_pct.max() * 0.8, stats_text_up, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Box plot 2: Lowest prices (negative moves, relevant for SL)
ax2 = axes[1]
bp2 = ax2.boxplot([lowest_pct], vert=True, patch_artist=True, widths=0.6,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7),
                   medianprops=dict(color='darkred', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
ax2.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Starting price')
ax2.set_title(f'Lowest Price in {MAX_HORIZON}-Minute Window\n(For Stop Loss Tuning)', 
              fontsize=12, fontweight='bold')
ax2.set_ylabel('Price Change (%)')
ax2.set_xticklabels(['Downside'])
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

stats_text_down = (f"Mean: {lowest_pct.mean():.4f}%\n"
                   f"Median: {np.median(lowest_pct):.4f}%\n"
                   f"Q1: {np.percentile(lowest_pct, 25):.4f}%\n"
                   f"Q3: {np.percentile(lowest_pct, 75):.4f}%\n"
                   f"Std: {lowest_pct.std():.4f}%")
ax2.text(1.35, lowest_pct.max() * 0.8, stats_text_down, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/price_movement_tp_sl_tuning.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/price_movement_tp_sl_tuning.png")
plt.close()

# Create combined box plot showing both sides
fig, ax = plt.subplots(figsize=(12, 7))

box_data = [highest_pct, lowest_pct]
bp = ax.boxplot(box_data, labels=['Upside (TP Target)', 'Downside (SL Target)'], 
                vert=True, patch_artist=True, widths=0.6,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

colors = ['lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
ax.set_ylabel('Price Change (%)', fontsize=11, fontweight='bold')
ax.set_title(f'Bitcoin Price Movement Distribution\nHorizon: {MAX_HORIZON} minutes (Post-2018 Data)\nGuide for TP/SL Threshold Selection', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax.text(1, highest_pct.max() + (highest_pct.max() - highest_pct.min()) * 0.05,
        f"MAX: {highest_pct.max():.4f}%\nQ3: {np.percentile(highest_pct, 75):.4f}%",
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.text(2, lowest_pct.min() - (lowest_pct.max() - lowest_pct.min()) * 0.08,
        f"MIN: {lowest_pct.min():.4f}%\nQ1: {np.percentile(lowest_pct, 25):.4f}%",
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/combined_price_movement_boxplot.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/combined_price_movement_boxplot.png")
plt.close()

print("\n✓ Data visualization complete!")
