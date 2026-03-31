"""
Backtesting Engine
Purpose: Simulate trading strategy using model predictions and evaluate real P&L
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================
# HELPER FUNCTIONS
# ================================

def calculate_pnl(trade_type, entry_price, exit_price, commission):
    """Calculate P&L for a single trade"""
    if trade_type == 'BUY':
        gross_return = (exit_price - entry_price) / entry_price
    else:  # SELL
        gross_return = (entry_price - exit_price) / entry_price
    
    net_return = gross_return - (2 * commission)  # Commission on entry + exit
    return net_return

# ================================
# CONFIGURATION
# ================================

PREDICTIONS_PATH = "predictions_test.csv"
FEATURES_PATH = "features.csv"

# Trading parameters
ENTRY_TIMEOUT = 60  # minutes to hold trade if signal doesn't reverse
SLIPPAGE = 0.0001   # 0.01% per trade (bid-ask spread assumption)
COMMISSION = 0.001  # 0.1% per trade
POSITION_SIZE = 1.0 # BTC per trade (for P&L calculation)

# ================================
# LOAD DATA
# ================================

print("Loading predictions and features...")
predictions = pd.read_csv(PREDICTIONS_PATH)
features = pd.read_csv(FEATURES_PATH)

print(f"Predictions shape: {predictions.shape}")
print(f"Features shape: {features.shape}")

# Align predictions with OHLC data
# Get close prices for P&L calculation
close_prices = features['Close'].values[-len(predictions):]
timestamps = features['Timestamp'].values[-len(predictions):]

print(f"\nTest set size: {len(predictions)} records")
print(f"Close price range: {close_prices.min():.2f} - {close_prices.max():.2f}")

# ================================
# TRADE SIMULATION
# ================================

print("\n===== SIMULATING TRADES =====\n")

trades = []
position = None  # Current open position: {'type': 'BUY'|'SELL', 'entry_idx': idx, 'entry_price': price}

for idx in tqdm(range(len(predictions)), desc="Simulating trades"):
    decision = predictions.iloc[idx]['decision']
    current_price = close_prices[idx]
    
    # Close existing position logic
    if position is not None:
        time_in_trade = idx - position['entry_idx']
        
        # Exit condition 1: Timeout
        if time_in_trade >= ENTRY_TIMEOUT:
            exit_price = current_price * (1 - SLIPPAGE) if position['type'] == 'BUY' else current_price * (1 + SLIPPAGE)
            pnl = calculate_pnl(position['type'], position['entry_price'], exit_price, COMMISSION)
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': idx,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'return': pnl,
                'reason': 'timeout'
            })
            position = None
        
        # Exit condition 2: Opposite signal
        elif (position['type'] == 'BUY' and decision == 'SELL') or (position['type'] == 'SELL' and decision == 'BUY'):
            exit_price = current_price * (1 - SLIPPAGE) if position['type'] == 'BUY' else current_price * (1 + SLIPPAGE)
            pnl = calculate_pnl(position['type'], position['entry_price'], exit_price, COMMISSION)
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': idx,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'return': pnl,
                'reason': 'reversal'
            })
            position = None
    
    # Enter new position
    if decision != 'NO_TRADE' and position is None:
        entry_price = current_price * (1 + SLIPPAGE) if decision == 'BUY' else current_price * (1 - SLIPPAGE)
        position = {
            'type': decision,
            'entry_idx': idx,
            'entry_price': entry_price
        }

# Close final position if still open
if position is not None:
    exit_price = close_prices[-1] * (1 - SLIPPAGE) if position['type'] == 'BUY' else close_prices[-1] * (1 + SLIPPAGE)
    pnl = calculate_pnl(position['type'], position['entry_price'], exit_price, COMMISSION)
    trades.append({
        'entry_idx': position['entry_idx'],
        'exit_idx': len(predictions) - 1,
        'type': position['type'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'return': pnl,
        'reason': 'final'
    })

# ================================
# BACKTEST ANALYSIS
# ================================

print("\n===== BACKTEST RESULTS =====\n")

if len(trades) == 0:
    print("No trades executed!")
else:
    trades_df = pd.DataFrame(trades)
    
    # Basic stats
    total_trades = len(trades)
    winning_trades = (trades_df['return'] > 0).sum()
    losing_trades = (trades_df['return'] <= 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['return'] <= 0]['return'].mean() if losing_trades > 0 else 0
    
    total_pnl = trades_df['return'].sum()
    
    buy_trades = trades_df[trades_df['type'] == 'BUY']
    sell_trades = trades_df[trades_df['type'] == 'SELL']
    
    print(f"Total trades executed: {total_trades}")
    print(f"  BUY trades: {len(buy_trades)}")
    print(f"  SELL trades: {len(sell_trades)}")
    print(f"\nWinning trades: {winning_trades} ({win_rate*100:.2f}%)")
    print(f"Losing trades: {losing_trades}")
    print(f"\nAverage win: {avg_win:.6f} BTC ({avg_win*100:.4f}%)")
    print(f"Average loss: {avg_loss:.6f} BTC ({avg_loss*100:.4f}%)")
    print(f"\nTotal P&L: {total_pnl:.6f} BTC ({total_pnl*100:.4f}%)")
    print(f"Cumulative return: {total_pnl*POSITION_SIZE:.4f} BTC")
    
    if winning_trades > 0:
        profit_factor = -total_pnl if losing_trades == 0 else abs(trades_df[trades_df['return'] > 0]['return'].sum() / trades_df[trades_df['return'] <= 0]['return'].sum())
        print(f"Profit factor: {profit_factor:.2f}")
    
    # Strategy comparison
    print(f"\nBUY strategy: {len(buy_trades)} trades, {(buy_trades['return'] > 0).sum()} wins, {(buy_trades['return'].sum()*100):.4f}% return")
    print(f"SELL strategy: {len(sell_trades)} trades, {(sell_trades['return'] > 0).sum()} wins, {(sell_trades['return'].sum()*100):.4f}% return")
    
    # Drawdown analysis
    cumulative_returns = trades_df['return'].cumsum()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    print(f"\nMax drawdown: {max_drawdown:.6f} BTC ({max_drawdown*100:.4f}%)")
    
    # Sharpe ratio (approximate)
    if len(trades_df) > 1:
        returns_std = trades_df['return'].std()
        sharpe = (trades_df['return'].mean() / returns_std * np.sqrt(252*24*60)) if returns_std > 0 else 0  # Annualized
        print(f"Sharpe ratio (annualized): {sharpe:.2f}")

# ================================
# VISUALIZATION
# ================================

print("\nGenerating visualization...")

if len(trades) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cumulative P&L
    ax1 = axes[0, 0]
    cumulative = trades_df['return'].cumsum().values
    ax1.plot(cumulative, linewidth=2, color='blue')
    ax1.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax1.set_title('Cumulative P&L Over Trades', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Cumulative Return (BTC)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Plot 2: Win/Loss distribution
    ax2 = axes[0, 1]
    returns = trades_df['return'].values
    ax2.hist(returns, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('P&L Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Return per Trade (BTC)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: BUY vs SELL performance
    ax3 = axes[1, 0]
    categories = ['BUY', 'SELL']
    buy_pnl = buy_trades['return'].sum() if len(buy_trades) > 0 else 0
    sell_pnl = sell_trades['return'].sum() if len(sell_trades) > 0 else 0
    ax3.bar(categories, [buy_pnl, sell_pnl], color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax3.set_title('Total P&L by Signal Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Return (BTC)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([buy_pnl, sell_pnl]):
        ax3.text(i, v, f'{v*100:.2f}%', ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
    
    # Plot 4: Trade duration distribution
    ax4 = axes[1, 1]
    durations = trades_df['exit_idx'] - trades_df['entry_idx']
    ax4.hist(durations, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Duration (minutes)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: backtest_results.png")
    plt.close()

# ================================
# SAVE DETAILED TRADE LOG
# ================================

if len(trades) > 0:
    trades_df.to_csv('backtest_trade_log.csv', index=False)
    print("✓ Saved: backtest_trade_log.csv")

print("\n✓ Backtesting complete!")
