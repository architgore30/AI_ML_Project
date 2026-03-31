"""
Backtesting Engine
Purpose: Simulate trading strategy using model predictions and evaluate real P&L
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ================================
# HELPER FUNCTIONS
# ================================

# (Helper functions removed - logic now inline for clarity)

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "models-4"
PREDICTIONS_PATH = MODEL_PATH + "/predictions_test.csv"
FEATURES_PATH = "features.csv"

# Trading parameters
ENTRY_TIMEOUT = 60  # minutes to hold trade if signal doesn't reverse
SLIPPAGE = 0.0001   # 0.01% per trade (bid-ask spread assumption)
COMMISSION = 0.001  # 0.1% per trade
INITIAL_BALANCE = 100000.0  # Starting USD balance (all deployed to BTC)

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

print("\n===== SIMULATING TRADES (LONG-ONLY, SPOT TRADING) =====\n")

trades = []
position = None  # Current open position: {'entry_idx': idx, 'entry_price': price, 'btc_amount': amount}

# Portfolio tracking
cash_balance = INITIAL_BALANCE
btc_holdings = 0.0

for idx in tqdm(range(len(predictions)), desc="Simulating trades"):
    decision = predictions.iloc[idx]['decision']
    current_price = close_prices[idx]
    
    # Check if we should exit current position
    if position is not None and btc_holdings > 0:
        time_in_trade = idx - position['entry_idx']
        should_exit = False
        exit_reason = None
        
        # Exit condition 1: Timeout (hold for ENTRY_TIMEOUT minutes)
        if time_in_trade >= ENTRY_TIMEOUT:
            should_exit = True
            exit_reason = 'timeout'
        
        # Exit condition 2: SELL signal (exit long position)
        elif decision == 'SELL':
            should_exit = True
            exit_reason = 'sell_signal'
        
        # Execute exit
        if should_exit:
            exit_price = current_price * (1 - SLIPPAGE)
            
            # Sell all BTC holdings
            cash_from_sale = btc_holdings * exit_price * (1 - COMMISSION)
            pnl_usd = cash_from_sale - (position['btc_amount'] * position['entry_price'] * (1 + COMMISSION))
            pnl_pct = (pnl_usd / (position['btc_amount'] * position['entry_price'] * (1 + COMMISSION))) * 100
            
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': idx,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'btc_traded': btc_holdings,
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct,
                'reason': exit_reason
            })
            
            cash_balance += cash_from_sale
            btc_holdings = 0.0
            position = None
    
    # Enter new BUY position if we have no holdings
    if btc_holdings == 0 and decision == 'BUY' and position is None:
        entry_price = current_price * (1 + SLIPPAGE)
        
        # Buy as much BTC as possible with all available cash
        btc_amount = (cash_balance / (entry_price * (1 + COMMISSION)))
        
        if btc_amount > 0:
            # Spend cash to buy BTC
            cash_spent = btc_amount * entry_price * (1 + COMMISSION)
            cash_balance -= cash_spent
            btc_holdings = btc_amount
            
            position = {
                'entry_idx': idx,
                'entry_price': entry_price,
                'btc_amount': btc_amount
            }

# Close final position if still open
if position is not None and btc_holdings > 0:
    exit_price = close_prices[-1] * (1 - SLIPPAGE)
    cash_from_sale = btc_holdings * exit_price * (1 - COMMISSION)
    pnl_usd = cash_from_sale - (position['btc_amount'] * position['entry_price'] * (1 + COMMISSION))
    pnl_pct = (pnl_usd / (position['btc_amount'] * position['entry_price'] * (1 + COMMISSION))) * 100
    
    trades.append({
        'entry_idx': position['entry_idx'],
        'exit_idx': len(predictions) - 1,
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'btc_traded': btc_holdings,
        'pnl_usd': pnl_usd,
        'pnl_pct': pnl_pct,
        'reason': 'end_of_data'
    })
    
    cash_balance += cash_from_sale
    btc_holdings = 0.0

# ================================
# BACKTEST ANALYSIS
# ================================

print("\n===== BACKTEST RESULTS =====\n")

# Final portfolio value
final_portfolio_value = cash_balance + (btc_holdings * close_prices[-1])
total_return = final_portfolio_value - INITIAL_BALANCE
total_return_pct = (total_return / INITIAL_BALANCE) * 100

print(f"Initial balance: ${INITIAL_BALANCE:,.2f}")
print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
print(f"Total P&L: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Final cash: ${cash_balance:,.2f}")
print(f"Final BTC holdings: {btc_holdings:.6f} BTC")
if btc_holdings > 0:
    print(f"BTC value at close: ${btc_holdings * close_prices[-1]:,.2f}")

if len(trades) == 0:
    print("\nNo trades executed!")
else:
    trades_df = pd.DataFrame(trades)
    
    # Basic stats
    total_trades = len(trades)
    winning_trades = (trades_df['pnl_usd'] > 0).sum()
    losing_trades = (trades_df['pnl_usd'] <= 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win_usd = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
    avg_loss_usd = trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd'].mean() if losing_trades > 0 else 0
    
    total_pnl_usd = trades_df['pnl_usd'].sum()
    
    print(f"\nTotal trades executed: {total_trades}")
    print(f"  All trades are LONG (buy and hold until SELL signal or timeout)")
    print(f"\nWinning trades: {winning_trades} ({win_rate*100:.2f}%)")
    print(f"Losing trades: {losing_trades}")
    print(f"\nAverage win: ${avg_win_usd:,.2f} ({trades_df[trades_df['pnl_usd'] > 0]['pnl_pct'].mean()*100:.4f}%)")
    print(f"Average loss: ${avg_loss_usd:,.2f} ({trades_df[trades_df['pnl_usd'] <= 0]['pnl_pct'].mean()*100:.4f}%)")
    print(f"\nTotal P&L from trades: ${total_pnl_usd:,.2f} ({total_pnl_usd*100/INITIAL_BALANCE:.2f}% of initial)")
    
    if winning_trades > 0:
        profit_factor = -total_pnl_usd if losing_trades == 0 else abs(trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum() / trades_df[trades_df['pnl_usd'] <= 0]['pnl_usd'].sum())
        print(f"Profit factor: {profit_factor:.2f}")
    
    # Drawdown analysis
    cumulative_usd = trades_df['pnl_usd'].cumsum()
    running_max = cumulative_usd.cummax()
    drawdown_usd = cumulative_usd - running_max
    max_drawdown_usd = drawdown_usd.min()
    max_drawdown_pct = (max_drawdown_usd / INITIAL_BALANCE) * 100
    
    print(f"\nMax drawdown: ${max_drawdown_usd:,.2f} ({max_drawdown_pct:.2f}%)")
    
    # Sharpe ratio (approximate, per trade)
    if len(trades_df) > 1:
        returns_std = trades_df['pnl_usd'].std()
        mean_return = trades_df['pnl_usd'].mean()
        sharpe = (mean_return / returns_std * np.sqrt(252*24*60)) if returns_std > 0 else 0  # Annualized approximation
        print(f"Sharpe ratio (approximate): {sharpe:.2f}")

# ================================
# VISUALIZATION
# ================================

print("\nGenerating visualization...")

if len(trades) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cumulative P&L in USD
    ax1 = axes[0, 0]
    cumulative_usd = trades_df['pnl_usd'].cumsum().values
    portfolio_usd = cumulative_usd + INITIAL_BALANCE
    ax1.plot(portfolio_usd, linewidth=2, color='blue', label='Portfolio Value')
    ax1.axhline(y=INITIAL_BALANCE, color='red', linestyle='--', linewidth=1, label='Initial Balance')
    ax1.fill_between(range(len(portfolio_usd)), INITIAL_BALANCE, portfolio_usd, alpha=0.3, color='blue')
    ax1.set_title('Cumulative Portfolio Value', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Win/Loss distribution (USD)
    ax2 = axes[0, 1]
    returns_usd = trades_df['pnl_usd'].values
    ax2.hist(returns_usd, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('P&L Distribution ($)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('P&L per Trade ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Win rate over time (cumulative wins)
    ax3 = axes[1, 0]
    cumulative_wins = (trades_df['pnl_usd'] > 0).cumsum()
    win_rate_over_time = cumulative_wins / (np.arange(len(cumulative_wins)) + 1)
    ax3.plot(win_rate_over_time, linewidth=2, color='green', label='Win Rate')
    ax3.axhline(y=win_rate, color='red', linestyle='--', linewidth=1, label=f'Overall: {win_rate*100:.1f}%')
    ax3.set_title('Win Rate Over Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_xlabel('Trade #')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format as percentage on y-axis
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    # Plot 4: Trade duration distribution
    ax4 = axes[1, 1]
    durations = trades_df['exit_idx'] - trades_df['entry_idx']
    ax4.hist(durations, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Duration (minutes)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_PATH}/backtest_results.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {MODEL_PATH}/backtest_results.png")
    plt.close()

# ================================
# SAVE DETAILED TRADE LOG
# ================================

if len(trades) > 0:
    trades_df.to_csv(MODEL_PATH + '/backtest_trade_log.csv', index=False)
    print(f"✓ Saved: {MODEL_PATH}backtest_trade_log.csv")

print("\n✓ Backtesting complete!")
