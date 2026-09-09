import { TrendingUp, TrendingDown, Layers, Shield, Crosshair } from 'lucide-react';

export default function OpenTrades({ trades }) {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={16} color="var(--accent)" />
          ACTIVE INSTITUTIONAL POSITIONS
        </div>
        <div className="card-tag">{trades.length} / 2 ACTIVE</div>
      </div>
      
      <div className="card-body" style={{ flex: 1, padding: 0, overflowX: 'auto' }}>
        {trades.length === 0 ? (
          <div style={{ display: 'flex', height: '180px', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontFamily: 'Space Mono', fontSize: '11px', gap: '8px' }}>
            <Shield size={22} color="var(--muted)" />
            <span>No active open positions — Scanning market for 15M POI setups...</span>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Type</th>
                <th>Volume</th>
                <th>Entry Price</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th style={{ textAlign: 'right' }}>Floating P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((pos) => {
                const action = (pos.type || pos.action || 'BUY').toUpperCase();
                const isBuy = action === 'BUY';
                const lot = pos.volume !== undefined ? pos.volume : (pos.lot_size !== undefined ? pos.lot_size : 0.01);
                const entryPrice = pos.price_open !== undefined ? pos.price_open : (pos.entry_price !== undefined ? pos.entry_price : 0.0);
                const profit = pos.profit !== undefined ? pos.profit : (pos.pnl !== undefined ? pos.pnl : 0.0);
                const isJpyOrGold = pos.symbol?.toUpperCase().includes('JPY') || pos.symbol?.toUpperCase().includes('XAU') || pos.symbol?.toUpperCase().includes('GOLD');
                
                const formatPrice = (val) => {
                  if (val === undefined || val === null || val === 0) return '—';
                  return typeof val === 'number' ? val.toFixed(isJpyOrGold ? 2 : 5) : val;
                };

                return (
                  <tr key={pos.ticket} className="slide-in">
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Crosshair size={14} color="var(--accent)" />
                        <span style={{ fontWeight: 800, color: '#fff', fontFamily: 'Syne', fontSize: '13px' }}>{pos.symbol}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${isBuy ? 'badge-buy' : 'badge-sell'}`} style={{ gap: '4px' }}>
                        {isBuy ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                        {action}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text)', fontFamily: 'Space Mono', fontWeight: 700 }}>{lot} Lot</td>
                    <td style={{ color: 'var(--text2)', fontFamily: 'Space Mono' }}>{formatPrice(entryPrice)}</td>
                    <td style={{ color: 'var(--sell)', fontFamily: 'Space Mono' }}>{formatPrice(pos.sl)}</td>
                    <td style={{ color: 'var(--buy)', fontFamily: 'Space Mono' }}>{formatPrice(pos.tp)}</td>
                    <td style={{ textAlign: 'right', fontWeight: 800, fontSize: '14px', fontFamily: 'Space Mono' }} className={profit >= 0 ? 'value-positive' : 'value-negative'}>
                      {profit >= 0 ? '+' : ''}${profit.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
