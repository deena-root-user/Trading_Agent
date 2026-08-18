import { useState } from 'react';
import { ShieldAlert, AlertOctagon } from 'lucide-react';

export default function KillSwitch({ onKill, paused, onPause, onResume }) {
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [closedCount, setClosedCount] = useState(null);

  const handleTriggerKill = async () => {
    setLoading(true);
    try {
      const data = await onKill();
      setClosedCount(data.closed || 0);
      setShowConfirm(false);
      setTimeout(() => setClosedCount(null), 5000);
    } catch (err) {
      console.error(err);
      alert('Kill switch trigger failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card fade-in" style={{ 
      borderColor: 'rgba(239, 68, 68, 0.3)',
      boxShadow: '0 0 25px rgba(239, 68, 68, 0.03)'
    }}>
      <div className="card-header" style={{ 
        background: 'rgba(239, 68, 68, 0.02)', 
        borderBottom: '1px solid rgba(239, 68, 68, 0.15)' 
      }}>
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--sell)' }}>
          <ShieldAlert size={14} style={{ filter: 'drop-shadow(0 0 4px var(--sell))' }} />
          EMERGENCY SYSTEM HALT
        </div>
        <div className="card-tag" style={{ borderColor: 'rgba(239, 68, 68, 0.2)', color: 'var(--sell)' }}>Critical</div>
      </div>
      
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '20px' }}>
        <div>
          <p style={{ fontSize: '12px', color: 'var(--text2)', lineHeight: '1.6', marginBottom: '16px' }}>
            Activating the Emergency Kill Switch will instantly close all open market positions on MT5, cancel pending orders, and halt the neural trading engine.
          </p>
          <button 
            className="btn btn-danger" 
            style={{ width: '100%', justifyContent: 'center', gap: '8px', padding: '12px', fontWeight: 800, borderRadius: 'var(--radius-sm)' }}
            onClick={() => setShowConfirm(true)}
          >
            <AlertOctagon size={14} />
            ACTIVATE SYSTEM KILL SWITCH
          </button>
        </div>

        {closedCount !== null && (
          <div style={{ 
            padding: '10px 14px', 
            background: 'rgba(239, 68, 68, 0.08)', 
            border: '1px solid var(--sell)', 
            borderRadius: 'var(--radius-sm)', 
            color: 'var(--sell)', 
            fontSize: '11px', 
            fontFamily: 'Space Mono',
            fontWeight: 700 
          }}>
            🚨 EMERGENCY CLOSE SUCCESSFUL: TERMINATED {closedCount} ACTIVE POSITIONS.
          </div>
        )}

        {showConfirm && (
          <div className="modal-overlay">
            <div className="modal fade-in">
              <div className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--sell)' }}>
                <AlertOctagon size={20} />
                CONFIRM DEVASTATING PROTOCOL
              </div>
              <div className="modal-desc" style={{ color: 'var(--text2)', fontSize: '13px', lineHeight: '1.6', marginTop: '10px' }}>
                This is a critical production operation. Proceeding will trigger the following events:
                <br /><br />
                1. <strong>Instantly close</strong> all active market positions on MT5.
                <br />
                2. Purge all queued execution orders.
                <br />
                3. Transmit priority Telegram alert.
                <br /><br />
                Confirm you want to execute this safety protocol.
              </div>
              <div className="modal-actions">
                <button 
                  className="btn" 
                  onClick={() => setShowConfirm(false)} 
                  disabled={loading}
                  style={{ background: 'rgba(255,255,255,0.02)', borderColor: 'rgba(255,255,255,0.06)', color: 'var(--text)' }}
                >
                  Cancel
                </button>
                <button 
                  className="btn btn-danger" 
                  onClick={handleTriggerKill} 
                  disabled={loading}
                  style={{ borderRadius: 'var(--radius-sm)' }}
                >
                  {loading ? 'Halted...' : 'YES, FORCE CLOSE ALL'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
