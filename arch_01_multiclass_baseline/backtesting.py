import json

import matplotlib.pyplot as plt

import pipeline_config as config
from pipeline_utils import ensure_directories, simulate_backtest


def main():
    ensure_directories()

    predictions_path = config.OUTPUTS_DIR / "predictions_decisions.csv"
    prediction_df = __import__("pandas").read_csv(predictions_path)
    summary, trades_df, equity_df = simulate_backtest(prediction_df)

    trade_log_path = config.OUTPUTS_DIR / "backtest_trade_log.csv"
    summary_path = config.OUTPUTS_DIR / "backtest_summary.json"
    chart_path = config.OUTPUTS_DIR / "backtest_results.png"

    trades_df.to_csv(trade_log_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    if not equity_df.empty:
        ax1.plot(equity_df["equity"].to_numpy(), linewidth=2)
    ax1.set_title("Equity Curve")
    ax1.set_xlabel("Bar")
    ax1.set_ylabel("Equity")
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    if not trades_df.empty:
        ax2.hist(trades_df["pnl_usd"].to_numpy(), bins=30, alpha=0.8)
    ax2.set_title("Trade PnL Distribution")
    ax2.set_xlabel("PnL USD")
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    if not equity_df.empty:
        running_max = equity_df["equity"].cummax()
        drawdown = ((equity_df["equity"] / running_max) - 1.0) * 100.0
        ax3.plot(drawdown.to_numpy(), color="red", linewidth=2)
    ax3.set_title("Drawdown %")
    ax3.set_xlabel("Bar")
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    if not trades_df.empty:
        ax4.plot(trades_df["holding_bars"].to_numpy(), linewidth=2)
    ax4.set_title("Holding Bars")
    ax4.set_xlabel("Trade #")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved trade log to {trade_log_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
