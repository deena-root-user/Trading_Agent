export default function StatusBar({ status, onPause, onResume }) {
  if (!status) return (
    <div className="card fade-in" style={{ padding: '16px 24px' }}>
      <span style={{ fontFamily: 'Space Mono', fontSize: '11px', color: 'var(--text2)', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div className="dot dot-warn" />
        INITIALIZING NEURAL TELEMETRY CORE...
      </span>
    </div>
  )

  const items = [
    { label: 'Neural Agent', value: status.paused ? 'PAUSED' : 'RUNNING', dot: status.paused ? 'dot-warn' : 'dot-online' },
    { label: 'Execution Engine', value: status.pro_trader_mode ? 'PRO TRADER (4-TF SMC)' : (status.auto_scalp_mode ? 'AUTO-SCALP' : 'STANDARD'), dot: 'dot-online' },
    { label: 'MT5 Terminal', value: status.mt5_connected ? 'CONNECTED' : 'DISCONNECTED', dot: status.mt5_connected ? 'dot-online' : 'dot-error' },
    { label: 'Neural Model', value: status.model || 'plutus-latest', dot: 'dot-online' },
    { label: 'Active Session', value: status.in_session ? (status.session || 'London').toUpperCase() : 'CLOSED', dot: status.in_session ? 'dot-online' : 'dot-warn' },
    { label: 'Trading Pairs', value: (status.pairs || []).join(' · ').toUpperCase(), dot: '' },
    { label: 'Base Volume', value: `${status.lot_size?.toFixed(2) || '0.01'} LOT`, dot: '' },
  ]

  return (
    <div className="fade-in" style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
      background: 'var(--surface)',
      backdropFilter: 'blur(24px) saturate(140%)',
      WebkitBackdropFilter: 'blur(24px) saturate(140%)',
      border: '1px solid var(--border2)',
      borderRadius: 'var(--radius)',
      overflow: 'hidden',
      boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    }}>
      {items.map((item, i) => (
        <div key={i} style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px 20px',
          borderRight: i < items.length - 1 ? '1px solid var(--border2)' : 'none',
          borderBottom: '1px solid var(--border2)',
          transition: 'background var(--transition)',
        }}
        className="status-item-hover"
        >
          {item.dot && <div className={`dot ${item.dot}`} style={{ marginTop: '2px' }} />}
          <div>
            <div style={{ 
              fontFamily: 'Space Mono', 
              fontSize: '8px', 
              letterSpacing: '1.5px', 
              color: 'var(--text2)', 
              textTransform: 'uppercase', 
              marginBottom: '3px',
              fontWeight: 700 
            }}>
              {item.label}
            </div>
            <div style={{ 
              fontFamily: 'Space Mono', 
              fontSize: '11px', 
              color: '#fff', 
              fontWeight: 700,
              letterSpacing: '0.5px' 
            }}>
              {item.value}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
