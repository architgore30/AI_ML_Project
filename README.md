# Bitcoin Trend Detection Trading System

**Status**: Active Development | **Phase**: Model Iteration & Regime Analysis

A production-grade machine learning system for detecting Bitcoin price trends on 1-minute OHLC data. Uses triple-barrier event-based labeling to generate high-quality trading signals, then trains gradient boosting models to predict buy/sell opportunities with risk-managed uncertainty filtering.

**Not a school project.** This is a real trading signal generator encoding risk management directly into the label generation logic.

---

## Quick Start

```bash
# 1. Generate labels on post-2018 Bitcoin data
python labelling.py

# 2. Engineer 35+ features (momentum, volatility, technical indicators)
python features.py

# 3. Train XGBoost models with asymmetric class weights
python model.py

# 4. Inspect data characteristics and identify regime issues
python data_visualization.py
```

---

## Architecture

### Data Pipeline

- **Input**: `dataset.csv` (7.5M rows of 1-minute Bitcoin OHLCV, 2012-2027)
- **Regime Filter**: Post-2018 only (~4.3M rows) to focus on modern market behavior
- **Labels**: Triple-barrier event detection (BUY, SELL, IDK)
- **Features**: 35 engineered indicators (raw + technical)
- **Models**: Two separate GradientBoostingClassifiers (BUY probability, SELL probability)
- **Output**: Uncertainty-filtered trading signals with adjustable thresholds

### Labeling Strategy (Triple-Barrier)

**Philosophy**: Risk-first, intentional asymmetry favoring caution.

| Label | Trigger | Interpretation |
|-------|---------|-----------------|
| **BUY** | Price hits +0.8% first | Strong uptrend forms within 60 min |
| **SELL** | Price hits -0.5% first | Downside risk detected within 60 min |
| **IDK** | Neither threshold hit | Market too choppy, no clear signal |

**Parameters** (tuned v2):
- `TP` (take profit): +0.8% (was +1.5%)
- `SL` (stop loss): -0.5% (was -0.7%)
- `MAX_HORIZON`: 60 minutes (was 30)

**Expected Distribution**: ~8-10% BUY, ~20% SELL, ~70% IDK

### Feature Engineering (35 features)

**Raw Transformations** (18):
- Returns (1m, 5m SMA, 10m SMA)
- High-low ranges and close positioning
- Moving averages (5, 10, 20 period)
- Volume metrics (SMA, ratio vs mean)
- Volatility measurements

**Technical Indicators** (17):
- RSI (5, 10, 14 period)
- MACD (12/26 + 9-period signal)
- Bollinger Bands (20-period, width & position)
- ATR (Average True Range, 14-period)
- Stochastic (%K, %D)
- CCI (Commodity Channel Index, 20-period)
- Directional Indicators (DI+, DI-, 14-period)

**Design Principle**: All features are backward-looking only (no future data leakage).

### Model Training

**Approach**: Two independent GradientBoostingClassifier models.

```python
# Class weighting: strong emphasis on rare positive cases
buy_weight = 1 / buy_ratio  # Amplify minority
sell_weight = 1 / sell_ratio
sample_weight = np.where(y == 1, weight, 1.0)  # Rare cases get boost
```

**Key hyperparameters**:
- `max_depth`: 5
- `learning_rate`: 0.1
- `subsample`: 0.8
- `n_estimators`: 100
- `warm_start`: True (incremental training with progress bars)

**Output**: Two independent probability scores (0-1).

**Decision Logic** (Uncertainty Filter):
```python
if buy_prob > 0.7 and sell_prob < 0.3:    TRADE BUY
elif sell_prob > 0.7 and buy_prob < 0.3:  TRADE SELL
else:                                      NO_TRADE
```

---

## Current Status

### Latest Results (Post-2018 Data + Aggressive Class Weights) ✅ BREAKTHROUGH

**Test Set (866k samples, post-2018 only)**:
- BUY Model ROC-AUC: 0.7768 | Recall: **47%** | Precision: 21% | F1: 0.29
- SELL Model ROC-AUC: 0.7463 | Recall: 73% | Precision: 30% | F1: 0.43
- **Trade generation: 240,966 signals (27.82%)** - realistic frequency
- **BUY signals: 1,286** (was 0 before!) ✓
- **SELL signals: 239,680** (controlled from 875k flood)
- **NO_TRADE: 625,119** (uncertainty filter working well)

**The Fix That Worked**: Removed pre-2018 data (old regime, 17.5% zero-volume samples) + applied aggressive class weighting. BUY recall jumped from 3% → 47%.

### Key Findings

**Regime Analysis** (data_visualization.py):
- 2012-2018: Higher volume (mean 7) but unstable, early Bitcoin era
- 2018+: Lower volume (mean 3) but modern market structure
- **17.5% of all samples have zero volume** → impossible to generate TP/SL signals
- Post-2018 data is the challenging regime; model should focus here

**Feature Importance** (XGBoost ranking):
1. Bollinger Bands (upper, lower, width) - 54% combined
2. ATR-14 (volatility) - 30%
3. Volume SMA - 10%
4. MACD indicators - 5%
5. Moving averages - 2%

### Known Issues

1. ✅ **SOLVED: BUY signal generation** - Now producing 1,286 BUY predictions vs zero before
2. ✅ **SOLVED: Trade frequency flood** - Down from 63% to 27.82% - realistic
3. ⚠️ **Precision still low**: BUY 21%, SELL 30% (must validate in backtesting whether profitable)
4. ⚠️ **Class imbalance remains**: BUY labels still rare even in post-2018 data

### Active Hypotheses

- Stronger class weights → Push model to predict more BUY signals ✓ In progress
- Post-2018 data only → Remove old-era distortion ✓ In progress
- Lower decision thresholds (0.6/0.4 vs 0.7/0.3) → Next test
- Alternative classifier (neural net with sigmoid) → Future

---

## Project Structure

```
.
├── dataset.csv                      # Raw Bitcoin OHLCV data (7.5M rows)
├── labelling.py                     # Triple-barrier label generation
├── features.py                      # 35+ feature engineering
├── model.py                         # XGBoost training & evaluation
├── data_visualization.py            # Dataset analysis (volume patterns, regimes)
├── models/                          # Trained model storage
│   ├── xgboost_buy_model.joblib
│   ├── xgboost_sell_model.joblib
│   └── feature_names.joblib
├── visualizations/                  # Analysis plots
│   ├── volume_analysis.png
│   ├── volume_by_period.png
│   └── xgboost_results.png
├── labeled_data.csv                 # Output from labelling.py
├── features.csv                     # Output from features.py
├── predictions_test.csv             # Model predictions on test set
├── notes                            # Development diary & decisions
├── .github/copilot-instructions.md  # Workspace AI context
└── README.md                        # This file
```

---

## Development Conventions

### Code Style
- Use pandas for data manipulation, numpy for numerical operations
- Use `.values` for performance on large arrays (7.5M rows)
- Use tqdm for progress tracking
- Configuration parameters at top of each script
- Use `np.int8` for label arrays to optimize memory

### Data Handling
- Always call `df.dropna()` before processing
- Time-based train/test split only (first 80%, last 20%)
- **NEVER shuffle time series data**
- Document data subsetting (e.g., `df = df.tail(200000) # for testing`)
- Class weights handle imbalance intentionally

### Quant Principles
- **No data leakage**: Training features never use future information
- **Class imbalance intentional**: SELL > BUY due to asymmetric thresholds (protective bias)
- **Expect 55-65% accuracy max**: Markets inherently ~50% random
- **Real edge**: Feature quality + uncertainty filtering > raw accuracy
- **Backtest everything**: Paper metrics misleading; profits are ground truth

---

## Configuration & Tuning

### Labeling Thresholds (labelling.py, lines 14-16)

```python
TP = 0.008       # +0.8% upward move for BUY (lower = more sensitive)
SL = 0.005       # -0.5% downward move for SELL (lower = earlier warning)
MAX_HORIZON = 60 # minutes to detect threshold (higher = longer trends)
```

**Tuning Guide**:
- Too many trades? Increase TP (e.g., 0.01) or MAX_HORIZON
- Too few trades? Decrease TP (e.g., 0.005) or decrease MAX_HORIZON
- Want more BUY vs SELL? Tighten SL (e.g., -0.003)

### Decision Thresholds (model.py, line ~180)

```python
if buy_proba > 0.7 and sell_proba < 0.3:  # Lower thresholds = more trades
```

**Tuning Guide**:
- Generating too many SELL signals? Raise SELL threshold (e.g., 0.8)
- Getting zero BUY signals? Lower BUY threshold (e.g., 0.5)
- Prefer fewer, higher-conviction trades? Raise both to (0.8/0.2)

### Class Weight Factor (model.py, line ~94)

```python
WEIGHT_FACTOR = 1.0  # Can tune: 1.0 (default), 2.0 (more emphasis), 5.0 (extreme)
```

**Effect**: Higher = more emphasis on getting rare BUY signals right, at cost of false positives.

---

## Roadmap

### Completed ✅
- [x] Triple-barrier labeling with 3-label system
- [x] 35+ feature engineering (raw + technical indicators)
- [x] XGBoost training with class weights
- [x] Data regime analysis (pre/post-2018)
- [x] Model persistence (joblib serialization)
- [x] Evaluation metrics (ROC-AUC, precision/recall, feature importance)

### In Progress 🟡
- [ ] Post-2018 data filtering & retraining
- [ ] Aggressive class weight tuning
- [ ] Decision threshold optimization
- [ ] Validation on recent volatile data

### Planned ☐
- [ ] Backtesting infrastructure (P&L, Sharpe ratio, drawdown)
- [ ] Real-time inference pipeline
- [ ] Neural network variant with sigmoid outputs (instead of XGBoost)
- [ ] Risk management layer (position sizing, stop-loss enforcement)
- [ ] Deployment (cloud API, webhook integration)

---

## Known Gotchas

### Data Quality
- **17.5% zero-volume samples**: No trading activity, impossible to generate valid signals
- **Timestamp column**: Unix seconds (not datetime strings)
- **Column naming**: Use uppercase (Timestamp, Open, High, Low, Close, Volume)

### Model Training
- **Never shuffle**: Use chronological train/test split only
- **Data leakage**: All features must look backward only
- **Class imbalance**: Expected and intentional; use class weights
- **Memory constraints**: Full 7.5M rows with features may require 16GB+ RAM

### Performance Interpretation
- **50-65% accuracy is strong** for market prediction
- **Real edge from uncertainty filtering**, not raw prediction accuracy
- **Backtest results trump paper metrics**: Validation set performance is often misleading

---

## Support & Questions

- For system architecture questions, see [Copilot Instructions](.github/copilot-instructions.md)
- For design decisions, see [Development Notes](notes)
- For data insights, run `python data_visualization.py`

---

**Last Updated**: March 31, 2026  
**Data Coverage**: 2012-2027 (7.5M 1-minute bars, actively filtered to 2018+)  
**Regime Focus**: Modern Bitcoin (post-2018 market structure)
