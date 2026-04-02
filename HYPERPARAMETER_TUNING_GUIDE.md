# Hyperparameter Tuning Strategy for Weak Hardware (i5 CPU)

## Problem
- Full grid search: 450 combinations x 2 models (BUY+SELL) x n_estimators each
- On i5 CPU: Can take 8-12+ hours to complete
- Not practical for iterative development

## Solution: Hardware-Aware Tuning

The `hyperparameter_tuning.py` script now automatically detects CPU cores and chooses strategy:

### LITE Search (Weak Hardware: ≤4 cores)
- **Parameter combinations**: 60 (2x2x2x2 instead of 6x5x3x5)
- **Data sample**: 25,000 rows (50% of full)
- **Estimated time**: 30-45 minutes on i5
- **Purpose**: Fast exploration for best parameter ranges

```python
n_estimators: [50, 150]           # Down from 6 values
max_depth: [2, 5]                 # Down from 5 values
learning_rate: [0.05, 0.1]        # Down from 3 values
subsample: [0.6, 1.0]             # Down from 5 values
```

### FULL Search (Standard/Strong Hardware: >4 cores)
- **Parameter combinations**: 450
- **Data sample**: 50,000 rows
- **Estimated time**: 2-4 hours on modern GPU/CPU
- **Purpose**: Comprehensive tuning for production

## GPU/CPU Automatic Fallback

Script detects GPU availability and switches devices:

```
Device Detection Flow:
  GPU CUDA available? 
    ├─ YES → Use gpu_hist (tree_method='gpu_hist', device='cuda')
    └─ NO  → Use hist (tree_method='hist', device='cpu')

During Training Error:
  GPU error during training?
    ├─ YES → Fall back to CPU automatically
    └─ Continue with all remaining combinations on CPU
```

## Workflow for i5 CPU

### Stage 1: LITE Search (Fast Exploration)
```bash
python hyperparameter_tuning.py
# Runs automatically on 60 combinations + 25k samples
# Output: tuning_results.csv with ranked parameters
```

### Stage 2: Understand Patterns
Review the top 10 results to understand:
- Which `max_depth` values work best? (2 or 5?)
- Which `learning_rate` is better? (0.05 or 0.1?)
- Do more estimators always help?

### Stage 3: Manual Fine-Tuning
Edit `LITE_SEARCH = False` in hyperparameter_tuning.py to expand:
```python
LITE_SEARCH = False  # Switch to FULL grid
```

Or run once more with expanded LITE parameters based on Stage 2:
```python
PARAM_GRID = {
    'n_estimators': [100, 150, 200],    # Expand around best range
    'max_depth': [3, 4, 5],
    'learning_rate': [0.08, 0.1, 0.12],
    'subsample': [0.6, 0.8, 1.0]
}
```

### Stage 4: Final Training
Train the final model on FULL dataset:
1. Copy best parameters from tuning results
2. Update `model.py` with these hyperparameters
3. Run `python model.py` on full dataset (7.5M rows)
4. Run `python backtest.py` for real profitability metrics

## Recommendations by Hardware

### i3/i5 (2-4 cores)
- Always use LITE_SEARCH
- Run Stage 1-2 first (30min)
- Manually expand grid if needed (Stage 3)
- Skip FULL grid search to save time

### i7/i9 (6-8+ cores)
- Can run FULL grid search directly
- Adjust `SAMPLE_SIZE = 100_000` for better signal
- GPU if available will speed up 3-5x

### GPU Available (NVIDIA CUDA)
- All grids run 3-5x faster
- Can do FULL search + larger sample simultaneously
- Auto falls back to CPU if GPU memory exhausted

## Parameter Interpretation

After tuning, interpret results:

**n_estimators**: More isn't always better
- If best result is at edge (350/400), expand upward
- If plateau appears around 100-150, stop there (diminishing returns)

**max_depth**: Shallow vs Deep trade-off
- Shallow (1-3): Faster, less overfit, but may underfit
- Deep (5-7): Better accuracy on training, but expensive

**learning_rate**: Speed of convergence
- Lower (0.01-0.05): Smoother but slower
- Higher (0.1): Faster but may oscillate
- Recommendation: Use 0.05-0.1 range for most cases

**subsample**: Row sampling per tree
- High (0.8-1.0): More data per tree, less regularization
- Low (0.2-0.4): More regularization, may underfit
- Recommendation: 0.6-1.0 for most trading datasets

## Expected Timing

| Hardware | Search Type | Time | Sample Size |
|----------|-------------|------|-------------|
| i5 2-core | LITE | 30-45 min | 25k |
| i5 4-core | LITE | 20-30 min | 25k |
| i7 6-core | FULL | 1.5-2 hrs | 50k |
| GPU (T4) | FULL | 20-30 min | 50k |
| GPU Titan | FULL | 5-10 min | 100k |

## Optimization Tips

1. **Reduce complexity** if tuning takes >45 mins on weak hardware
   ```python
   PARAM_GRID['n_estimators'] = [50, 100]  # Test only 2 values
   ```

2. **Sample data wisely** - More data = better patterns but slower
   ```python
   SAMPLE_SIZE = 20_000  # Cut further for very weak hardware
   ```

3. **Use validation** after tuning to prevent overfitting
   - Train on first 70%, validate on next 15%, test on last 15%
   - Current code uses 80/20 split - good for most cases

4. **Monitor CPU usage** during tuning
   ```bash
   # Linux/Mac: watch -n 1 'ps aux | grep python'
   # Windows: Task Manager → Performance tab
   ```

5. **Save intermediate results** in case of interruption
   - tuning_results.csv is written after each combination
   - Safe to Ctrl+C and restart (results cumulative)
