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

print("\n✓ Data visualization complete!")
