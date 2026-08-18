import { AlertTriangle, ShieldCheck, ShieldAlert, BrainCircuit } from 'lucide-react';

export default function DecisionFeed({ decisions }) {
  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BrainCircuit size={14} color="var(--accent)" />
          LLM DECISION LOG
        </div>
        <div className="card-tag">Plutus Core</div>
      </div>
      
      <div className="card-body" style={{ flex: 1, padding: 0, maxHeight: '430px', overflowY: 'auto' }}>
        {decisions.length === 0 ? (
          <div style={{ display: 'flex', height: '180px', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontFamily: 'Space Mono', fontSize: '11px' }}>
            No decisions logged yet
          </div>
        ) : (
          <div style={{ padding: '16px' }}>
            {decisions.map((dec) => {
              const isActionable = dec.action === 'BUY' || dec.action === 'SELL';
              
              return (
                <div key={dec.id} style={{ 
                  background: 'rgba(255, 255, 255, 0.01)', 
                  border: '1px solid rgba(255, 255, 255, 0.03)', 
                  borderRadius: 'var(--radius-sm)', 
                  padding: '16px', 
                  marginBottom: '12px',
                  boxShadow: 'inset 0 1px 1px rgba(255, 255, 255, 0.01)'
                }} className="slide-in">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                    <span style={{ fontFamily: 'Space Mono', fontWeight: 700, fontSize: '13px', color: '#fff' }}>{dec.symbol}</span>
                    <span className={`badge ${dec.action === 'BUY' ? 'badge-buy' : dec.action === 'SELL' ? 'badge-sell' : 'badge-hold'}`}>
                      {dec.action}
                    </span>
                    <span style={{ fontSize: '10px', color: 'var(--text2)', marginLeft: 'auto', fontFamily: 'Space Mono', opacity: 0.8 }}>
                      {new Date(dec.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  {isActionable && (
                    <div style={{ 
                      display: 'flex', 
                      flexWrap: 'wrap',
                      gap: '12px', 
                      fontFamily: 'Space Mono', 
                      fontSize: '11px', 
                      marginBottom: '8px', 
                      color: 'var(--text2)',
                      background: 'rgba(255, 255, 255, 0.01)',
                      padding: '6px 10px',
                      borderRadius: '4px',
                      border: '1px solid rgba(255, 255, 255, 0.02)'
                    }}>
                      <span>Conf: <strong style={{ color: '#fff' }}>{(dec.confidence * 100).toFixed(0)}%</strong></span>
                      <span>Entry: <strong style={{ color: '#fff' }}>{dec.entry}</strong></span>
                      <span>SL: <strong style={{ color: '#fff' }}>{dec.sl}</strong></span>
                      <span>TP: <strong style={{ color: '#fff' }}>{dec.tp}</strong></span>
                    </div>
                  )}

                  {dec.pattern && (
                    <div style={{ fontSize: '11px', color: 'var(--accent2)', fontFamily: 'Space Mono', marginBottom: '6px', fontWeight: 700 }}>
                      Pattern: <span>{dec.pattern}</span>
                    </div>
                  )}

                  <div style={{ fontSize: '12px', color: 'var(--text)', lineHeight: '1.6', marginBottom: '10px' }}>
                    {dec.reasoning}
                  </div>

                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px', 
                    borderTop: '1px solid rgba(255, 255, 255, 0.04)', 
                    paddingTop: '10px',
                    fontSize: '11px',
                    fontFamily: 'Space Mono'
                  }}>
                    {dec.risk_passed ? (
                      <>
                        <ShieldCheck size={14} color="var(--buy)" />
                        <span style={{ color: 'var(--buy)', fontWeight: 700 }}>RISK OK — ORDER DISPATCHED</span>
                      </>
                    ) : dec.action === 'HOLD' ? (
                      <>
                        <ShieldCheck size={14} color="var(--hold)" />
                        <span style={{ color: 'var(--hold)', fontWeight: 700 }}>STANDBY — MONITORING</span>
                      </>
                    ) : (
                      <>
                        <ShieldAlert size={14} color="var(--sell)" />
                        <span style={{ color: 'var(--sell)', fontWeight: 700 }}>RISK VETO: {dec.block_reason}</span>
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
