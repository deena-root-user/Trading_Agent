import { TrendingUp, TrendingDown, Layers } from 'lucide-react';

export default function OpenTrades({ trades }) {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={14} color="var(--accent)" />
          ACTIVE OPEN POSITIONS
        </div>
        <div className="card-tag">{trades.length} Active</div>
      </div>
      
      <div className="card-body" style={{ flex: 1, padding: 0, overflowX: 'auto' }}>
        {trades.length === 0 ? (
          <div style={{ display: 'flex', height: '180px', alignItems: 'center', justifycontent: 'center', justifyContent: 'center', color: 'var(--text2)', fontFamily: 'Space Mono', fontSize: '11px' }}>
            No active open positions on MT5
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Type</th>
                <th>Lot</th>
                <th>Entry</th>
                <th>SL</th>
                <th>TP</th>
                <th style={{ textAlign: 'right' }}>Profit</th>
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
                      <span style={{ fontWeight: 700, color: '#fff' }}>{pos.symbol}</span>
                    </td>
                    <td>
                      <span className={`badge ${isBuy ? 'badge-buy' : 'badge-sell'}`} style={{ gap: '4px' }}>
                        {isBuy ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                        {action}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text)' }}>{lot}</td>
                    <td style={{ color: 'var(--text2)' }}>{formatPrice(entryPrice)}</td>
                    <td style={{ color: 'var(--text2)' }}>{formatPrice(pos.sl)}</td>
                    <td style={{ color: 'var(--text2)' }}>{formatPrice(pos.tp)}</td>
                    <td style={{ textAlign: 'right', fontWeight: 700 }} className={profit >= 0 ? 'value-positive' : 'value-negative'}>
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
