export default function PnlSummary({ pnl, status }) {
  const stats = [
    {
      label: 'Today P&L',
      value: pnl ? `${pnl.today_pnl >= 0 ? '+' : ''}$${pnl.today_pnl?.toFixed(2)}` : '—',
      color: !pnl ? 'var(--text)' : pnl.today_pnl >= 0 ? 'var(--buy)' : 'var(--sell)',
      sub: pnl ? (
        <div style={{ display: 'flex', gap: '6px' }}>
          <span className="stat-chip stat-chip-green">{pnl.today_wins} W</span>
          <span className="stat-chip stat-chip-red">{pnl.today_losses} L</span>
        </div>
      ) : null,
    },
    {
      label: 'Win Rate',
      value: pnl ? `${pnl.win_rate?.toFixed(1)}%` : '—',
      color: !pnl ? 'var(--text)' : pnl.win_rate >= 55 ? 'var(--buy)' : pnl.win_rate >= 40 ? 'var(--warn)' : 'var(--sell)',
      bar: pnl?.win_rate,
      barColor: pnl?.win_rate >= 55 ? 'var(--buy)' : pnl?.win_rate >= 40 ? 'var(--warn)' : 'var(--sell)',
      sub: pnl ? (
        <span className="stat-chip stat-chip-accent">{pnl.total_trades} TRADES</span>
      ) : null,
    },
    {
      label: 'Account Balance',
      value: pnl?.balance != null ? `$${pnl.balance.toFixed(2)}` : '—',
      color: 'var(--text)',
      sub: <span className="stat-chip">MT5 TERMINAL</span>,
    },
    {
      label: 'Equity State',
      value: pnl?.equity != null ? `$${pnl.equity.toFixed(2)}` : '—',
      color: !pnl?.equity || !pnl?.balance ? 'var(--text)'
        : pnl.equity >= pnl.balance ? 'var(--buy)' : 'var(--sell)',
      sub: pnl?.equity != null && pnl?.balance ? (
        <span className={`stat-chip ${pnl.equity >= pnl.balance ? 'stat-chip-green' : 'stat-chip-red'}`}>
          {pnl.equity >= pnl.balance ? '+' : ''}${(pnl.equity - pnl.balance).toFixed(2)} FLT
        </span>
      ) : null,
    },
  ]

  return (
    <div className="grid-2">
      {stats.map((stat, i) => (
        <div key={i} className="card fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between', minHeight: '120px', padding: '20px' }}>
            <div className="stat-block">
              <div className="stat-label">{stat.label}</div>
              <div className="stat-value" style={{ color: stat.color, fontSize: '20px' }}>
                {stat.value}
              </div>
            </div>
            
            <div style={{ marginTop: 'auto', width: '100%' }}>
              {stat.bar !== undefined && (
                <div className="progress-bar" style={{ margin: '8px 0 6px 0' }}>
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(stat.bar, 100)}%`, background: stat.barColor }}
                  />
                </div>
              )}
              {stat.sub && (
                <div className="stat-sub" style={{ marginTop: '6px' }}>
                  {stat.sub}
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
