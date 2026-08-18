import { useState, useEffect } from 'react';
import { Sliders, Check, AlertCircle } from 'lucide-react';

export default function LotSizeEditor({ currentLot, onUpdate }) {
  const [lotInput, setLotInput] = useState('');
  const [showConfirm, setShowConfirm] = useState(false);
  const [targetLot, setTargetLot] = useState(0);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [prevLot, setPrevLot] = useState(null);

  useEffect(() => {
    if (currentLot != null && currentLot !== prevLot && !isFocused && !showConfirm) {
      setLotInput(currentLot.toFixed(2));
      setPrevLot(currentLot);
    }
  }, [currentLot, prevLot, isFocused, showConfirm]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const val = parseFloat(lotInput);
    if (isNaN(val) || val <= 0 || val > 100) {
      alert('Please enter a valid lot size between 0.01 and 100');
      return;
    }
    setTargetLot(val);
    setShowConfirm(true);
  };

  const handleCancel = () => {
    setShowConfirm(false);
    if (currentLot != null) {
      setLotInput(currentLot.toFixed(2));
    }
  };

  const handleConfirm = async () => {
    setLoading(true);
    setMessage('');
    try {
      const res = await onUpdate(targetLot);
      if (res.success) {
        setMessage(`Lot size updated to ${targetLot}!`);
        setTimeout(() => setMessage(''), 3000);
        setShowConfirm(false);
        setPrevLot(targetLot);
      } else {
        alert(res.detail || 'Update failed');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to update lot size');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card fade-in">
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sliders size={14} color="var(--accent)" />
          LOT CONFIGURATION
        </div>
        <div className="card-tag">Risk Parameter</div>
      </div>
      
      <div className="card-body" style={{ padding: '20px' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px' }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '9px', color: 'var(--text2)', letterSpacing: '1.5px', textTransform: 'uppercase', display: 'block', marginBottom: '8px', fontFamily: 'Space Mono', fontWeight: 700 }}>
              Base Trading Volume (Lot)
            </label>
            <input
              className="input"
              type="number"
              step="0.01"
              value={lotInput}
              onChange={(e) => setLotInput(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              style={{ height: '39px' }}
            />
          </div>
          <button type="submit" className="btn btn-primary" style={{ alignSelf: 'flex-end', height: '39px', fontWeight: 700, borderRadius: 'var(--radius-sm)' }}>
            Update
          </button>
        </form>

        {message && (
          <div style={{ marginTop: '12px', fontSize: '11px', color: 'var(--buy)', display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'Space Mono' }}>
            <Check size={12} />
            {message}
          </div>
        )}

        {showConfirm && (
          <div className="modal-overlay">
            <div className="modal fade-in">
              <div className="modal-title" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <AlertCircle color="var(--warn)" size={20} />
                CONFIRM RISK ADJUSTMENT
              </div>
              <div className="modal-desc" style={{ color: 'var(--text2)', fontSize: '13px', lineHeight: '1.6', marginTop: '10px' }}>
                Are you sure you want to adjust the system lot size from <strong>{currentLot?.toFixed(2)}</strong> to <strong>{targetLot.toFixed(2)}</strong>?
                <br /><br />
                <span style={{ color: 'var(--warn)', fontWeight: 600 }}>⚠️ WARNING: All subsequent trades will immediately execute with this new volume setting. Ensure balance constraints are satisfied.</span>
              </div>
              
              <div className="modal-actions">
                <button 
                  className="btn" 
                  onClick={handleCancel} 
                  disabled={loading}
                  style={{ background: 'rgba(255,255,255,0.02)', borderColor: 'rgba(255,255,255,0.06)', color: 'var(--text)' }}
                >
                  Cancel
                </button>
                <button 
                  className="btn btn-warn" 
                  onClick={handleConfirm} 
                  disabled={loading}
                  style={{ borderRadius: 'var(--radius-sm)' }}
                >
                  {loading ? 'Updating...' : 'Yes, Confirm'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
