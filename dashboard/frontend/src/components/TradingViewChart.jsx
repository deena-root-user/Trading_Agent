import { useState, useEffect, useRef } from 'react'

export default function TradingViewChart({ symbol = 'XAUUSD' }) {
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [askingAi, setAskingAi] = useState(false)
  const abortControllerRef = useRef(null)

  const fetchAnalysis = async (signal) => {
    try {
      const res = await fetch(`/api/tradingview/analysis?symbol=${symbol}`, { signal })
      if (!res.ok) return
      const data = await res.json()
      setAnalysis(data)
      setLoading(false)
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Error fetching TradingView analysis:', e)
      }
    }
  }

  const askAiSuggestion = async () => {
    setAskingAi(true)
    try {
      const res = await fetch(`/api/tradingview/ai-suggestion?symbol=${symbol}`, { method: 'POST' })
      const data = await res.json()
      setAnalysis(data)
    } catch (e) {
      console.error('Error asking AI suggestion:', e)
    } finally {
      setAskingAi(false)
    }
  }

  useEffect(() => {
    let intervalId = null
    const controller = new AbortController()
    abortControllerRef.current = controller

    // On-demand fetch ONLY when this page/tab is active
    const startPolling = () => {
      fetchAnalysis(controller.signal)
      intervalId = setInterval(() => {
        if (document.visibilityState === 'visible') {
          fetchAnalysis(controller.signal)
        }
      }, 15000)
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden' && intervalId) {
        clearInterval(intervalId)
        intervalId = null
      } else if (document.visibilityState === 'visible' && !intervalId) {
        startPolling()
      }
    }

    startPolling()
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // CLEANUP: Immediately stop interval & abort any active backend request on tab unmount/navigate away
    return () => {
      if (intervalId) clearInterval(intervalId)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      controller.abort()
    }
  }, [symbol])

  const tvSymbol = `OANDA:${symbol.toUpperCase()}`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
      {/* Dynamic AI Move Suggestion Banner */}
      <div className="card-glass" style={{
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        borderLeft: '4px solid var(--primary)',
        background: 'rgba(15, 23, 42, 0.8)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: '300px' }}>
          <span style={{ fontSize: '20px' }}>🎯</span>
          <div>
            <span style={{ fontFamily: 'Space Mono', fontSize: '10px', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '1px' }}>
              CURRENT AI MOVE RECOMMENDATION
            </span>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#fff', marginTop: '2px' }}>
              {askingAi ? '⚡ Requesting live move suggestion from Ollama...' : (analysis?.move_suggestion || (loading ? 'Loading market analysis...' : 'Awaiting signal...'))}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Ask AI Button */}
          <button
            onClick={askAiSuggestion}
            disabled={askingAi}
            style={{
              background: askingAi ? 'rgba(99, 102, 241, 0.4)' : 'linear-gradient(135deg, #2563eb, #7c3aed)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 16px',
              fontSize: '12px',
              fontWeight: 700,
              fontFamily: 'Space Mono',
              cursor: askingAi ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 4px 14px rgba(37, 99, 235, 0.35)',
              transition: 'all 0.2s ease',
              whiteSpace: 'nowrap'
            }}
          >
            {askingAi ? '⏳ ASKING OLLAMA...' : '⚡ ASK AI SUGGESTION'}
          </button>

          {/* Mode Badge */}
          {analysis?.focus_mode && (
            <div style={{
              padding: '6px 14px',
              borderRadius: '20px',
              fontSize: '11px',
              fontWeight: 800,
              fontFamily: 'Space Mono',
              letterSpacing: '0.5px',
              background: analysis.focus_mode.active ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
              color: analysis.focus_mode.active ? '#ef4444' : '#10b981',
              border: `1px solid ${analysis.focus_mode.active ? '#ef4444' : '#10b981'}`
            }}>
              {analysis.focus_mode.active 
                ? `🔥 HIGH FOCUS MODE (${analysis.focus_mode.consecutive_losses} LOSSES)` 
                : '🟢 PRO TRADER MODE'}
            </div>
          )}
        </div>
      </div>

      {/* Main Container: TradingView Chart + Analysis HUD */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px', height: '680px' }}>
        {/* TradingView Embedded IFrame Widget */}
        <div className="card-glass" style={{ padding: 0, overflow: 'hidden', height: '100%' }}>
          <iframe
            title="TradingView Chart"
            src={`https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=${tvSymbol}&interval=5&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=STD%3BRSI%3BSTD%3BEMA%3BSTD%3BMACD&theme=dark&style=1&timezone=Etc%2FUTC`}
            style={{ width: '100%', height: '100%', border: 'none' }}
          />
        </div>

        {/* Real-time Technical Analysis HUD */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
          {/* AI Decision Panel */}
          <div className="card-glass" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text2)', textTransform: 'uppercase' }}>AI DECISION</span>
              <span className={`badge ${analysis?.ai_decision?.action === 'BUY' ? 'badge-buy' : (analysis?.ai_decision?.action === 'SELL' ? 'badge-sell' : 'badge-hold')}`}>
                {analysis?.ai_decision?.action || 'HOLD'}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontFamily: 'Space Mono', fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Entry Price:</span>
                <span style={{ color: '#fff', fontWeight: 700 }}>
                  {analysis?.ai_decision?.entry && analysis.ai_decision.entry > 0
                    ? analysis.ai_decision.entry.toFixed(2)
                    : (loading ? 'Loading...' : (analysis?.technical_analysis?.price?.close ? `${analysis.technical_analysis.price.close.toFixed(2)} (Market)` : 'Ref Market'))}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Take Profit (TP):</span>
                <span style={{ color: 'var(--buy)', fontWeight: 700 }}>
                  {analysis?.ai_decision?.tp && analysis.ai_decision.tp > 0
                    ? analysis.ai_decision.tp.toFixed(2)
                    : (loading ? 'Loading...' : (analysis?.technical_analysis?.support_resistance?.resistance_1 ? `${analysis.technical_analysis.support_resistance.resistance_1.toFixed(2)} (R1)` : 'Ref Target'))}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Stop Loss (SL):</span>
                <span style={{ color: 'var(--sell)', fontWeight: 700 }}>
                  {analysis?.ai_decision?.sl && analysis.ai_decision.sl > 0
                    ? analysis.ai_decision.sl.toFixed(2)
                    : (loading ? 'Loading...' : (analysis?.technical_analysis?.support_resistance?.support_1 ? `${analysis.technical_analysis.support_resistance.support_1.toFixed(2)} (S1)` : 'Ref Stop'))}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Confidence:</span>
                <span style={{ color: '#818cf8', fontWeight: 700 }}>{((analysis?.ai_decision?.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Support & Resistance Panel */}
          <div className="card-glass" style={{ padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text2)', textTransform: 'uppercase', marginBottom: '12px' }}>
              📈 SUPPORT & RESISTANCE
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontFamily: 'Space Mono', fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Resistance 2 (R2):</span>
                <span style={{ color: 'var(--sell)' }}>
                  {analysis?.technical_analysis?.support_resistance?.resistance_2
                    ? analysis.technical_analysis.support_resistance.resistance_2.toFixed(2)
                    : (loading ? 'Loading...' : 'Ref R2')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Resistance 1 (R1):</span>
                <span style={{ color: 'var(--sell)' }}>
                  {analysis?.technical_analysis?.support_resistance?.resistance_1
                    ? analysis.technical_analysis.support_resistance.resistance_1.toFixed(2)
                    : (loading ? 'Loading...' : 'Ref R1')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Pivot Level (P):</span>
                <span style={{ color: '#fbbf24', fontWeight: 700 }}>
                  {analysis?.technical_analysis?.support_resistance?.pivot
                    ? analysis.technical_analysis.support_resistance.pivot.toFixed(2)
                    : (loading ? 'Loading...' : 'Ref P')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Support 1 (S1):</span>
                <span style={{ color: 'var(--buy)' }}>
                  {analysis?.technical_analysis?.support_resistance?.support_1
                    ? analysis.technical_analysis.support_resistance.support_1.toFixed(2)
                    : (loading ? 'Loading...' : 'Ref S1')}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Support 2 (S2):</span>
                <span style={{ color: 'var(--buy)' }}>
                  {analysis?.technical_analysis?.support_resistance?.support_2
                    ? analysis.technical_analysis.support_resistance.support_2.toFixed(2)
                    : (loading ? 'Loading...' : 'Ref S2')}
                </span>
              </div>
            </div>
          </div>

          {/* Smart Money Concepts Panel */}
          <div className="card-glass" style={{ padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text2)', textTransform: 'uppercase', marginBottom: '12px' }}>
              ⚡ SMART MONEY CONCEPTS (SMC)
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontFamily: 'Space Mono', fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Fair Value Gap (FVG):</span>
                <span style={{ color: '#a78bfa', fontWeight: 700 }}>{analysis?.technical_analysis?.smart_money_concepts?.fvg_type || 'NONE'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>FVG Zone:</span>
                <span style={{ color: '#fff' }}>
                  {analysis?.technical_analysis?.smart_money_concepts?.fvg_bottom?.toFixed(2)} - {analysis?.technical_analysis?.smart_money_concepts?.fvg_top?.toFixed(2)}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Order Block (OB):</span>
                <span style={{ color: '#38bdf8', fontWeight: 700 }}>{analysis?.technical_analysis?.smart_money_concepts?.order_block_type || 'NONE'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>Structure:</span>
                <span style={{ color: '#34d399', fontWeight: 700 }}>{analysis?.technical_analysis?.smart_money_concepts?.market_structure || 'NEUTRAL'}</span>
              </div>
            </div>
          </div>

          {/* RSI & Moving Trend Panel */}
          <div className="card-glass" style={{ padding: '16px' }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--text2)', textTransform: 'uppercase', marginBottom: '12px' }}>
              📊 RSI & EMA MOVING TREND
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontFamily: 'Space Mono', fontSize: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>RSI (14):</span>
                <span style={{ color: '#fff' }}>{analysis?.technical_analysis?.rsi?.value} ({analysis?.technical_analysis?.rsi?.status})</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>EMA 50/200 Trend:</span>
                <span style={{ color: '#10b981', fontWeight: 700 }}>{analysis?.technical_analysis?.ema_trend?.trend}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text2)' }}>EMA 9/21 Cross:</span>
                <span style={{ color: '#fff' }}>{analysis?.technical_analysis?.ema_trend?.cross}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
