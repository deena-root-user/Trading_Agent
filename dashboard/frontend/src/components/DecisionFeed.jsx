import { AlertTriangle, ShieldCheck, ShieldAlert, BrainCircuit, Target, ArrowUpRight, ArrowDownRight, Layers } from 'lucide-react';

export default function DecisionFeed({ decisions }) {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BrainCircuit size={16} color="var(--accent)" />
          INSTITUTIONAL SMC TELEMETRY STREAM
        </div>
        <div className="card-tag">Pro Trader v2 Core</div>
      </div>
      
      <div className="card-body" style={{ flex: 1, padding: 0, maxHeight: '560px', overflowY: 'auto' }}>
        {decisions.length === 0 ? (
          <div style={{ display: 'flex', height: '220px', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontFamily: 'Space Mono', fontSize: '11px', gap: '10px' }}>
            <Layers size={24} color="var(--muted)" />
            <span>Awaiting telemetry signals from active market scan...</span>
          </div>
        ) : (
          <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {decisions.map((dec) => {
              const isActionable = dec.action === 'BUY' || dec.action === 'SELL';
              const hasPoiSetup = dec.entry > 0;
              const riskPts = Math.abs(dec.entry - dec.sl);
              
              return (
                <div key={dec.id} style={{ 
                  background: 'rgba(15, 23, 42, 0.65)', 
                  border: '1px solid rgba(255, 255, 255, 0.05)', 
                  borderRadius: 'var(--radius-sm)', 
                  padding: '16px', 
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)',
                  backdropFilter: 'blur(12px)'
                }} className="slide-in">
                  
                  {/* Top Bar */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                    <span style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '14px', color: '#fff', letterSpacing: '0.5px' }}>
                      {dec.symbol}
                    </span>
                    <span className={`badge ${dec.action === 'BUY' ? 'badge-buy' : dec.action === 'SELL' ? 'badge-sell' : 'badge-hold'}`}>
                      {dec.action === 'BUY' && <ArrowUpRight size={12} />}
                      {dec.action === 'SELL' && <ArrowDownRight size={12} />}
                      {dec.action}
                    </span>

                    {dec.pattern && (
                      <span className="glow-pill-accent" style={{ fontSize: '9px', fontFamily: 'Space Mono' }}>
                        {dec.pattern}
                      </span>
                    )}

                    <span style={{ fontSize: '10px', color: 'var(--text2)', marginLeft: 'auto', fontFamily: 'Space Mono' }}>
                      {new Date(dec.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                  </div>

                  {/* High-Tech POI Setup Details Box */}
                  {hasPoiSetup && (
                    <div className="poi-setup-box">
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'Space Mono', fontSize: '10px', fontWeight: 700, color: 'var(--accent)' }}>
                          <Target size={12} />
                          <span>UPCOMING POI ENTRY PARAMETERS</span>
                        </div>
                        <span className="glow-pill-green" style={{ fontSize: '9px' }}>
                          {dec.rr_ratio ? `${dec.rr_ratio.toFixed(2)} R:R` : '2.0+ R:R'}
                        </span>
                      </div>

                      <div className="poi-level-row">
                        <span style={{ color: 'var(--text2)' }}>📥 Limit Entry:</span>
                        <strong style={{ color: '#fff' }}>{dec.entry.toFixed(3)}</strong>
                      </div>
                      <div className="poi-level-row">
                        <span style={{ color: 'var(--text2)' }}>🛑 Stop Loss:</span>
                        <strong style={{ color: 'var(--sell)' }}>{dec.sl.toFixed(3)} ({riskPts.toFixed(2)} pts)</strong>
                      </div>
                      <div className="poi-level-row">
                        <span style={{ color: 'var(--text2)' }}>🎯 Take Profit:</span>
                        <strong style={{ color: 'var(--buy)' }}>{dec.tp.toFixed(3)}</strong>
                      </div>
                    </div>
                  )}

                  {/* Analysis Reasoning */}
                  <div style={{ fontSize: '12px', color: 'var(--text)', lineHeight: '1.5', margin: '8px 0 12px 0', fontFamily: 'Outfit' }}>
                    {dec.reasoning}
                  </div>

                  {/* Status Footer */}
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px', 
                    borderTop: '1px dashed rgba(255, 255, 255, 0.06)', 
                    paddingTop: '10px',
                    fontSize: '10px',
                    fontFamily: 'Space Mono'
                  }}>
                    {dec.risk_passed ? (
                      <>
                        <ShieldCheck size={14} color="var(--buy)" />
                        <span style={{ color: 'var(--buy)', fontWeight: 700 }}>VALIDATED & DISPATCHED</span>
                      </>
                    ) : dec.action === 'HOLD' ? (
                      <>
                        <ShieldCheck size={14} color="var(--hold)" />
                        <span style={{ color: 'var(--hold)', fontWeight: 700 }}>STANDBY — MONITORING POI</span>
                      </>
                    ) : (
                      <>
                        <ShieldAlert size={14} color="var(--sell)" />
                        <span style={{ color: 'var(--sell)', fontWeight: 700 }}>RISK GATE VETO: {dec.block_reason}</span>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
