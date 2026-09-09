import pandas as pd
import numpy as np

df = pd.read_csv('backtest_trades.csv')
output = []
output.append('=' * 70)
output.append('  PAXIS BACKTEST LOSS ANALYSIS REPORT')
output.append('=' * 70)

output.append('\n📊 STRATEGY PERFORMANCE BREAKDOWN')
output.append(str(df.groupby('strategy')['pnl_usd'].agg(
    Trades='count',
    WinRate=lambda x: f"{(x > 0).mean() * 100:.1f}%",
    TotalPnL=lambda x: f"${x.sum():+.2f}",
    AvgPnL=lambda x: f"${x.mean():+.2f}"
)))

output.append('\n📊 MARKET REGIME BREAKDOWN')
output.append(str(df.groupby('regime')['pnl_usd'].agg(
    Trades='count',
    WinRate=lambda x: f"{(x > 0).mean() * 100:.1f}%",
    TotalPnL=lambda x: f"${x.sum():+.2f}",
    AvgPnL=lambda x: f"${x.mean():+.2f}"
)))

losers = df[df['pnl_usd'] < 0].copy()
winners = df[df['pnl_usd'] > 0].copy()

output.append(f'\n📉 LOSING TRADES SUMMARY')
output.append(f'  Total Losing Trades:  {len(losers)} / {len(df)} ({len(losers)/len(df)*100:.1f}%)')
output.append(f'  Total USD Lost:        ${losers["pnl_usd"].sum():.2f}')
output.append(f'  Average Loss per Trade:${losers["pnl_usd"].mean():.2f}')
output.append(f'  Max Single Loss:       ${losers["pnl_usd"].min():.2f}')

output.append('\n📊 LOSSES BY CONFLUENCE SCORE')
losers['conf_bin'] = pd.cut(losers['confluence_score'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], labels=['<0.50', '0.50-0.60', '0.60-0.70', '0.70-0.80', '>0.80'])
output.append(str(losers.groupby('conf_bin', observed=False)['pnl_usd'].agg(
    LossCount='count',
    TotalLoss=lambda x: f"${x.sum():.2f}",
    AvgLoss=lambda x: f"${x.mean():.2f}"
)))

output.append('\n📊 LOSSES BY HOLDING DURATION (1M BARS)')
losers['hold_bin'] = pd.cut(losers['holding_bars'], bins=[-1, 5, 15, 30, 35, 1000], labels=['1-5m (Quick SL)', '6-15m', '16-30m', '31-35m (Stall)', '>35m'])
output.append(str(losers.groupby('hold_bin', observed=False)['pnl_usd'].agg(
    LossCount='count',
    TotalLoss=lambda x: f"${x.sum():.2f}",
    AvgLoss=lambda x: f"${x.mean():.2f}"
)))

df['time'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['time'].dt.hour
output.append('\n📊 WIN RATE & PNL BY TRADING HOUR (UTC)')
output.append(str(df.groupby('hour')['pnl_usd'].agg(
    Trades='count',
    WinRate=lambda x: f"{(x > 0).mean() * 100:.1f}%",
    TotalPnL=lambda x: f"${x.sum():+.2f}"
)))

output.append('\n🚨 TOP 15 WORST LOSSES RECORD')
worst = losers.sort_values('pnl_usd').head(15)
for idx, r in worst.iterrows():
    output.append(f"Time: {r['timestamp']} | {r['direction']} | Entry: {r['entry_price']:.2f} | SL: {r['sl_price']:.2f} | Exit: {r['exit_price']:.2f} | PnL: ${r['pnl_usd']:+.2f} ({r['pnl_r']:+.2f}R) | Strat: {r['strategy']} | Conf: {r['confluence_score']:.2f} | Hold: {r['holding_bars']}m")

with open('loss_report.txt', 'w') as f:
    f.write('\n'.join(output))
print("Wrote loss_report.txt successfully")
