import { useState, useEffect } from 'react';
import { Layers, Play, Square, Check, RefreshCw, Zap, ShieldAlert, ShieldCheck } from 'lucide-react';

export default function TradingController({
  paused,
  currentPairs,
  disableRiskGate,
  onToggleRiskGate,
  onPause,
  onResume,
  fetchAll,
  autoScalpMode,
  onToggleAutoScalp,
  autoScalpSL,
  autoScalpTP,
  autoScalpCycle,
}) {
  const [selectedChips, setSelectedChips] = useState([]);
  const [customInput, setCustomInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');
  const [prevPairsStr, setPrevPairsStr] = useState('');

  // Auto-Scalp fields and sync states
  const [scalpSL, setScalpSL] = useState('2.00');
  const [scalpTP, setScalpTP] = useState('1.00');
  const [scalpCycle, setScalpCycle] = useState('3');

  const [prevSL, setPrevSL] = useState(null);
  const [prevTP, setPrevTP] = useState(null);
  const [prevCycle, setPrevCycle] = useState(null);

  const [isSLFocused, setIsSLFocused] = useState(false);
  const [isTPFocused, setIsTPFocused] = useState(false);
  const [isCycleFocused, setIsCycleFocused] = useState(false);

  const [scalpSaving, setScalpSaving] = useState(false);
  const [scalpSuccessMsg, setScalpSuccessMsg] = useState('');

  // Default preset chips
  const presets = ['XAUUSD', 'EURUSD', 'GBPUSD', 'AUDJPY'];

  useEffect(() => {
    if (currentPairs) {
      const currentPairsStr = [...currentPairs].sort().join(',').toUpperCase();
      if (currentPairsStr !== prevPairsStr) {
        const upperPairs = currentPairs.map(p => p.toUpperCase().trim());
        setSelectedChips(presets.filter(p => upperPairs.includes(p)));
        const nonPresets = upperPairs.filter(p => !presets.includes(p));
        setCustomInput(nonPresets.join(', '));
        setPrevPairsStr(currentPairsStr);
      }
    }
  }, [currentPairs, prevPairsStr]);

  useEffect(() => {
    if (autoScalpSL !== undefined && autoScalpSL !== prevSL && !isSLFocused) {
      setScalpSL(autoScalpSL.toString());
      setPrevSL(autoScalpSL);
    }
  }, [autoScalpSL, prevSL, isSLFocused]);

  useEffect(() => {
    if (autoScalpTP !== undefined && autoScalpTP !== prevTP && !isTPFocused) {
      setScalpTP(autoScalpTP.toString());
      setPrevTP(autoScalpTP);
    }
  }, [autoScalpTP, prevTP, isTPFocused]);

  useEffect(() => {
    if (autoScalpCycle !== undefined && autoScalpCycle !== prevCycle && !isCycleFocused) {
      setScalpCycle(autoScalpCycle.toString());
      setPrevCycle(autoScalpCycle);
    }
  }, [autoScalpCycle, prevCycle, isCycleFocused]);

  // Handle Preset Chip Click
  const handleChipClick = (symbol) => {
    setSelectedChips(prev => {
      const next = prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol];
      return next;
    });
  };

  // Handle Save Pairs
  const handleSavePairs = async (e) => {
    e.preventDefault();
    setSaving(true);
    setSuccessMsg('');

    // Combine preset chips and custom inputs
    const customPairs = customInput
      .split(',')
      .map(p => p.trim().toUpperCase())
      .filter(p => p.length > 0);

    const allPairs = Array.from(new Set([...selectedChips, ...customPairs]));

    if (allPairs.length === 0) {
      alert('You must select or input at least one trading pair.');
      setSaving(false);
      return;
    }

    const pairsStr = allPairs.join(',');

    try {
      const res = await fetch('/api/config/pairs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trading_pairs: pairsStr }),
      });
      const data = await res.json();
      if (data.success) {
        setSuccessMsg('Active pairs successfully updated!');
        setTimeout(() => setSuccessMsg(''), 3000);
        setPrevPairsStr(allPairs.sort().join(',').toUpperCase());
        if (fetchAll) fetchAll();
      } else {
        alert(data.detail || 'Failed to update trading pairs');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to update trading pairs');
    } finally {
      setSaving(false);
    }
  };

  // Handle Save Scalp Config
  const handleSaveScalpConfig = async (e) => {
    if (e) e.preventDefault();
    setScalpSaving(true);
    setScalpSuccessMsg('');
    const sl = parseFloat(scalpSL);
    const tp = parseFloat(scalpTP);
    const cycle = parseInt(scalpCycle);

    if (isNaN(sl) || sl <= 0) {
      alert('Please enter a valid stop loss target (> 0)');
      setScalpSaving(false);
      return;
    }
    if (isNaN(tp) || tp <= 0) {
      alert('Please enter a valid take profit target (> 0)');
      setScalpSaving(false);
      return;
    }
    if (isNaN(cycle) || cycle < 1) {
      alert('Please enter a valid scan cycle in minutes (>= 1)');
      setScalpSaving(false);
      return;
    }

    try {
      const data = await onToggleAutoScalp(true, {
        sl_usd: sl,
        tp_usd: tp,
        cycle_minutes: cycle
      });
      if (data && data.success) {
        setScalpSuccessMsg('Scalp configuration updated and activated!');
        setTimeout(() => setScalpSuccessMsg(''), 3000);
        setPrevSL(sl);
        setPrevTP(tp);
        setPrevCycle(cycle);
      } else {
        alert((data && data.detail) || 'Failed to update Auto-Scalp configuration');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to save Auto-Scalp configuration');
    } finally {
      setScalpSaving(false);
    }
  };

  return (
    <div className="card fade-in">
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={14} color="var(--accent)" />
          TRADING OPERATIONS CENTER
        </div>
        <div className="card-tag">Neural Gateway</div>
      </div>
      
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {/* Real-time Status Banner */}
        <div style={{
          padding: '18px 24px',
          background: paused ? 'rgba(239, 68, 68, 0.03)' : 'rgba(16, 185, 129, 0.03)',
          border: paused ? '1px solid rgba(239, 68, 68, 0.15)' : '1px solid rgba(16, 185, 129, 0.15)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          transition: 'all var(--transition)',
        }}>
          <div>
            <div style={{
              fontSize: '8px',
              fontFamily: 'Space Mono',
              color: 'var(--text2)',
              letterSpacing: '1.5px',
              textTransform: 'uppercase',
              marginBottom: '4px',
              fontWeight: 700,
            }}>
              Core Engine Status
            </div>
            <div style={{
              fontSize: '18px',
              fontFamily: 'Syne',
              fontWeight: 800,
              color: paused ? 'var(--sell)' : 'var(--buy)',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}>
              <div className={`dot ${paused ? 'dot-error' : 'dot-online'}`} />
              {paused ? 'HALTED' : 'ACTIVE'}
            </div>
          </div>
          
          <div style={{
            fontSize: '11px',
            fontFamily: 'Space Mono',
            color: 'var(--text2)',
            textAlign: 'right',
            background: 'rgba(255, 255, 255, 0.015)',
            border: '1px solid rgba(255, 255, 255, 0.03)',
            padding: '8px 14px',
            borderRadius: '6px',
          }}>
            Pairs: <span style={{ color: '#fff', fontWeight: 700 }}>{(currentPairs || []).join(', ').toUpperCase()}</span>
          </div>
        </div>

        {/* Start / Stop Control Buttons */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '16px',
        }}>
          <button
            onClick={onResume}
            disabled={!paused}
            className="btn btn-success"
            style={{
              padding: '14px',
              fontSize: '12px',
              fontWeight: 800,
              fontFamily: 'Syne',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            <Play size={14} fill="#fff" />
            START AGENT
          </button>
          
          <button
            onClick={onPause}
            disabled={paused}
            className="btn btn-danger"
            style={{
              padding: '14px',
              fontSize: '12px',
              fontWeight: 800,
              fontFamily: 'Syne',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            <Square size={12} fill="#fff" />
            STOP AGENT
          </button>
        </div>

        {/* Auto-Scalp Systems Control Panel */}
        <div style={{
          padding: '20px',
          background: autoScalpMode ? 'rgba(99, 102, 241, 0.03)' : 'rgba(255, 255, 255, 0.015)',
          border: autoScalpMode ? '1px solid rgba(99, 102, 241, 0.2)' : '1px solid rgba(255, 255, 255, 0.03)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          transition: 'all var(--transition)',
        }}>
          {/* Header & Main Auto-Scalp Toggle */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{
                fontSize: '8px',
                fontFamily: 'Space Mono',
                color: 'var(--text2)',
                letterSpacing: '1.5px',
                textTransform: 'uppercase',
                marginBottom: '3px',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                <Zap size={10} color="var(--accent)" />
                AUTONOMOUS SCALPING ENGINE
              </div>
              <div style={{
                fontSize: '14px',
                fontFamily: 'Syne',
                fontWeight: 800,
                color: autoScalpMode ? 'var(--accent2)' : 'var(--text2)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <div className={`dot ${autoScalpMode ? 'dot-online' : ''}`} style={{ background: autoScalpMode ? 'var(--accent)' : 'var(--muted)', boxShadow: autoScalpMode ? '0 0 10px var(--accent)' : 'none' }} />
                {autoScalpMode ? 'AUTO-SCALP: ACTIVE' : 'AUTO-SCALP: PAUSED'}
              </div>
            </div>
            
            <button
              onClick={() => {
                if (autoScalpMode) {
                  onToggleAutoScalp && onToggleAutoScalp(false);
                } else {
                  handleSaveScalpConfig();
                }
              }}
              disabled={scalpSaving}
              className={`btn ${autoScalpMode ? 'btn-danger' : 'btn-primary'}`}
              style={{
                padding: '8px 16px',
                fontSize: '10px',
                fontWeight: 700,
                fontFamily: 'Space Mono',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'all var(--transition)',
              }}
            >
              {scalpSaving ? 'SAVING...' : (autoScalpMode ? 'DEACTIVATE' : 'ACTIVATE')}
            </button>
          </div>

          {/* Interactive Parameters Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
            gap: '12px',
            background: 'rgba(7, 10, 19, 0.4)',
            padding: '16px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.02)',
            fontSize: '11px',
            fontFamily: 'Space Mono'
          }}>
            <div>
              <label style={{ color: 'var(--text2)', fontSize: '8px', display: 'block', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 700 }}>Fixed SL ($ USD / 0.01 lot)</label>
              <input
                className="input"
                type="number"
                step="0.01"
                min="0.01"
                value={scalpSL}
                onChange={(e) => setScalpSL(e.target.value)}
                onFocus={() => setIsSLFocused(true)}
                onBlur={() => setIsSLFocused(false)}
                style={{ width: '100%', height: '36px', fontSize: '12px', fontFamily: 'Space Mono' }}
              />
            </div>
            <div>
              <label style={{ color: 'var(--text2)', fontSize: '8px', display: 'block', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 700 }}>Fixed TP ($ USD / 0.01 lot)</label>
              <input
                className="input"
                type="number"
                step="0.01"
                min="0.01"
                value={scalpTP}
                onChange={(e) => setScalpTP(e.target.value)}
                onFocus={() => setIsTPFocused(true)}
                onBlur={() => setIsTPFocused(false)}
                style={{ width: '100%', height: '36px', fontSize: '12px', fontFamily: 'Space Mono' }}
              />
            </div>
            <div>
              <label style={{ color: 'var(--text2)', fontSize: '8px', display: 'block', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 700 }}>Scan Cycle (Minutes)</label>
              <input
                className="input"
                type="number"
                step="1"
                min="1"
                value={scalpCycle}
                onChange={(e) => setScalpCycle(e.target.value)}
                onFocus={() => setIsCycleFocused(true)}
                onBlur={() => setIsCycleFocused(false)}
                style={{ width: '100%', height: '36px', fontSize: '12px', fontFamily: 'Space Mono' }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={handleSaveScalpConfig}
                disabled={scalpSaving}
                className="btn btn-primary"
                style={{
                  height: '36px',
                  fontFamily: 'Space Mono',
                  fontSize: '10px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  width: '100%',
                }}
              >
                {scalpSaving ? (
                  <RefreshCw size={12} className="spin" />
                ) : (
                  <Check size={12} />
                )}
                Save Config
              </button>
            </div>
          </div>

          {scalpSuccessMsg && (
            <div style={{
              fontSize: '11px',
              color: 'var(--buy)',
              fontFamily: 'Space Mono',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginTop: '-4px',
              paddingLeft: '4px',
            }}>
              <Check size={12} />
              {scalpSuccessMsg}
            </div>
          )}

          {/* Conditional Auto-Scalp Risk Gate Bypass Toggle */}
          {autoScalpMode && (
            <div style={{
              padding: '12px 14px',
              background: disableRiskGate ? 'rgba(239, 68, 68, 0.04)' : 'rgba(16, 185, 129, 0.04)',
              border: disableRiskGate ? '1px solid rgba(239, 68, 68, 0.15)' : '1px solid rgba(16, 185, 129, 0.15)',
              borderRadius: '6px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              transition: 'all var(--transition)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {disableRiskGate ? <ShieldAlert size={14} color="var(--sell)" /> : <ShieldCheck size={14} color="var(--buy)" />}
                  <span style={{
                    fontSize: '11px',
                    fontFamily: 'Space Mono',
                    fontWeight: 700,
                    color: disableRiskGate ? 'var(--sell)' : 'var(--buy)'
                  }}>
                    {disableRiskGate ? 'SAFETY GATE: BYPASSED' : 'SAFETY GATE: ENFORCED'}
                  </span>
                </div>
                
                <button
                  onClick={() => onToggleRiskGate && onToggleRiskGate(!disableRiskGate)}
                  className={`btn ${disableRiskGate ? 'btn-success' : 'btn-danger'}`}
                  style={{
                    padding: '4px 10px',
                    fontSize: '9px',
                    fontWeight: 700,
                    fontFamily: 'Space Mono',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    transition: 'all var(--transition)',
                  }}
                >
                  {disableRiskGate ? 'ENFORCE SAFETY' : 'BYPASS SAFETY'}
                </button>
              </div>

              <div style={{ fontSize: '9px', color: 'var(--text2)', fontFamily: 'system-ui', lineHeight: 1.3 }}>
                {disableRiskGate ? (
                  <span style={{ color: 'var(--sell)' }}>
                    <strong>WARNING:</strong> 10 Institutional safety checks are completely bypassed for autonomous scalping execution! Early exits and immediate orders are processed upon LLM output.
                  </span>
                ) : (
                  <span>
                    <strong>PROTECTED:</strong> Full multi-timeframe trend alignment, spread checks, drawdown monitoring, and economic news blackouts are validated before any scalp is opened.
                  </span>
                )}
              </div>
            </div>
          )}

          {!autoScalpMode && (
            <div style={{ fontSize: '10px', color: 'var(--text2)', fontFamily: 'system-ui', lineHeight: 1.4 }}>
              <span>
                <strong>AUTO-SCALP:</strong> Specialized autonomous short-term scalper cycle. Uses M1, M5, M15 timeframes to open, manage, and close tight trades. Enable to start autonomous scalping.
              </span>
            </div>
          )}
        </div>

        <div className="divider" style={{ margin: '8px 0' }} />

        {/* Active Instrument Selector */}
        <div>
          <label style={{
            fontSize: '9px',
            color: 'var(--text2)',
            letterSpacing: '1.5px',
            textTransform: 'uppercase',
            display: 'block',
            marginBottom: '12px',
            fontFamily: 'Space Mono',
            fontWeight: 700,
          }}>
            Trading Instrument Target Matrix
          </label>

          {/* Quick preset chips */}
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '8px',
            marginBottom: '16px'
          }}>
            {presets.map(symbol => {
              const isSelected = selectedChips.includes(symbol);
              return (
                <button
                  key={symbol}
                  type="button"
                  onClick={() => handleChipClick(symbol)}
                  style={{
                    padding: '8px 16px',
                    fontSize: '11px',
                    fontFamily: 'Space Mono',
                    fontWeight: 700,
                    borderRadius: '30px',
                    background: isSelected ? 'rgba(99, 102, 241, 0.12)' : 'rgba(255, 255, 255, 0.015)',
                    color: isSelected ? 'var(--accent2)' : 'var(--text2)',
                    border: isSelected ? '1px solid var(--accent)' : '1px solid rgba(255, 255, 255, 0.03)',
                    boxShadow: isSelected ? '0 0 10px rgba(99, 102, 241, 0.15)' : 'none',
                    cursor: 'pointer',
                    transition: 'all var(--transition)',
                  }}
                >
                  {symbol}
                </button>
              );
            })}
          </div>

          {/* Custom pair input */}
          <form onSubmit={handleSavePairs} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                className="input"
                type="text"
                placeholder="Custom pairs e.g. AUDJPY, USDCAD, BTCUSD"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                style={{
                  flex: 1,
                  fontFamily: 'Space Mono',
                  fontSize: '12px',
                }}
              />
              <button
                type="submit"
                disabled={saving}
                className="btn btn-primary"
                style={{
                  height: '39px',
                  fontFamily: 'Space Mono',
                  fontSize: '11px',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                {saving ? (
                  <RefreshCw size={12} className="spin" />
                ) : (
                  <Check size={12} />
                )}
                Save Matrix
              </button>
            </div>
            
            {successMsg && (
              <div style={{
                fontSize: '11px',
                color: 'var(--buy)',
                fontFamily: 'Space Mono',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                marginTop: '6px',
              }}>
                <Check size={12} />
                {successMsg}
              </div>
            )}
          </form>
        </div>

      </div>
    </div>
  );
}
