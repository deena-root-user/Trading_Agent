import { useState, useEffect, useCallback } from 'react'
import StatusBar from './components/StatusBar'
import PnlSummary from './components/PnlSummary'
import EquityCurve from './components/EquityCurve'
import OpenTrades from './components/OpenTrades'
import DecisionFeed from './components/DecisionFeed'
import LotSizeEditor from './components/LotSizeEditor'
import KillSwitch from './components/KillSwitch'
import TradingController from './components/TradingController'
import TradingViewChart from './components/TradingViewChart'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/live`

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [status, setStatus] = useState(null)
  const [pnl, setPnl] = useState(null)
  const [equity, setEquity] = useState([])
  const [trades, setTrades] = useState([])
  const [decisions, setDecisions] = useState([])
  const [wsConnected, setWsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)

  // ── Fetch initial data ─────────────────────────────────────────────────────
  const fetchAll = useCallback(async () => {
    try {
      const [s, p, e, t, d] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/pnl/today').then(r => r.json()),
        fetch('/api/equity?limit=200').then(r => r.json()),
        fetch('/api/trades/open').then(r => r.json()),
        fetch('/api/decisions?limit=30').then(r => r.json()),
      ])
      setStatus(s)
      setPnl(p)
      setEquity(e.snapshots || [])
      setTrades(t.positions || [])
      setDecisions(d.decisions || [])
      setLastUpdate(new Date())
    } catch (err) {
      console.error('Fetch error:', err)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  // Refresh every 40 seconds
  useEffect(() => {
    const id = setInterval(fetchAll, 40000)
    return () => clearInterval(id)
  }, [fetchAll])

  // ── WebSocket live updates ─────────────────────────────────────────────────
  useEffect(() => {
    let ws, retryTimer

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL)

        ws.onopen = () => {
          setWsConnected(true)
          // Keep-alive ping
          const pingId = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send('ping')
          }, 20000)
          ws._pingId = pingId
        }

        ws.onmessage = ({ data }) => {
          try {
            const { type, data: payload } = JSON.parse(data)
            setLastUpdate(new Date())

            if (type === 'CONNECTED') {
              setStatus(prev => ({ ...prev, ...payload }))
            } else if (type === 'TRADE_OPEN') {
              setTrades(prev => [payload, ...prev])
              fetchAll()
            } else if (type === 'TRADE_CLOSE') {
              setTrades(prev => prev.filter(t => t.ticket !== payload.ticket))
              fetchAll()
            } else if (type === 'LLM_DECISION') {
              setDecisions(prev => [payload, ...prev.slice(0, 49)])
            } else if (type === 'EQUITY_UPDATE') {
              setEquity(prev => [...prev.slice(-199), payload])
              setPnl(prev => prev ? { ...prev, balance: payload.balance, equity: payload.equity } : {
                today_pnl: 0.0,
                today_wins: 0,
                today_losses: 0,
                total_trades: 0,
                win_rate: 0.0,
                balance: payload.balance,
                equity: payload.equity
              })
            } else if (type === 'AGENT_STATUS') {
              setStatus(prev => ({ ...prev, ...payload }))
            } else if (type === 'LOT_UPDATED') {
              setStatus(prev => ({ ...prev, lot_size: payload.new }))
            }
          } catch (_) {}
        }

        ws.onclose = () => {
          setWsConnected(false)
          clearInterval(ws._pingId)
          retryTimer = setTimeout(connect, 3000)
        }

        ws.onerror = () => ws.close()
      } catch (_) {
        retryTimer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      clearTimeout(retryTimer)
      if (ws) { clearInterval(ws._pingId); ws.close() }
    }
  }, [fetchAll])

  // ── Agent control actions ──────────────────────────────────────────────────
  const onPause  = async () => { await fetch('/api/control/pause',  { method: 'POST' }); fetchAll() }
  const onResume = async () => { await fetch('/api/control/resume', { method: 'POST' }); fetchAll() }
  const onToggleRiskGate = async (disable) => {
    const endpoint = disable ? '/api/control/risk-gate/disable' : '/api/control/risk-gate/enable'
    await fetch(endpoint, { method: 'POST' })
    fetchAll()
  }
  const onToggleAutoScalp = async (enable, config) => {
    const endpoint = enable ? '/api/control/auto-scalp/enable' : '/api/control/auto-scalp/disable'
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config || {})
    })
    const data = await res.json()
    fetchAll()
    return data
  }
  const onKill   = async () => {
    const res = await fetch('/api/control/kill', { method: 'POST' })
    const data = await res.json()
    setTrades([])
    fetchAll()
    return data
  }
  const onLotUpdate = async (newLot) => {
    const res = await fetch('/api/config/lot-size', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lot_size: newLot }),
    })
    const data = await res.json()
    fetchAll()
    return data
  }

  return (
    <div style={{ minHeight: '100vh', padding: '0', display: 'flex', flexDirection: 'column' }}>
      {/* Premium Institutional Header */}
      <header style={{
        background: 'rgba(10, 16, 30, 0.75)',
        backdropFilter: 'blur(24px) saturate(140%)',
        WebkitBackdropFilter: 'blur(24px) saturate(140%)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '0 32px',
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
        height: '70px',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        boxShadow: '0 4px 30px rgba(0, 0, 0, 0.4)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
            width: '32px',
            height: '32px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
          }}>
            <span style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: '18px', color: '#fff' }}>P</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{
              fontFamily: 'Syne',
              fontSize: '20px',
              fontWeight: 800,
              color: '#fff',
              letterSpacing: '-0.5px',
              lineHeight: 1.1,
            }}>
              PAXIS
            </span>
            <span style={{
              fontFamily: 'Space Mono',
              fontSize: '8px',
              letterSpacing: '2px',
              color: 'var(--text2)',
              textTransform: 'uppercase',
              fontWeight: 700,
            }}>INTELLIGENT TELEMETRY</span>
          </div>
        </div>

        {/* Tab Navigation Switcher */}
        <div style={{ display: 'flex', gap: '8px', marginLeft: '24px' }}>
          <button
            onClick={() => setActiveTab('dashboard')}
            style={{
              background: activeTab === 'dashboard' ? 'rgba(99, 102, 241, 0.25)' : 'transparent',
              border: activeTab === 'dashboard' ? '1px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.1)',
              color: activeTab === 'dashboard' ? '#fff' : 'var(--text2)',
              padding: '6px 16px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontFamily: 'Space Mono',
              fontSize: '11px',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            📊 TELEMETRY DASHBOARD
          </button>
          <button
            onClick={() => setActiveTab('tradingview')}
            style={{
              background: activeTab === 'tradingview' ? 'rgba(99, 102, 241, 0.25)' : 'transparent',
              border: activeTab === 'tradingview' ? '1px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.1)',
              color: activeTab === 'tradingview' ? '#fff' : 'var(--text2)',
              padding: '6px 16px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontFamily: 'Space Mono',
              fontSize: '11px',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            📈 TRADINGVIEW ANALYSIS CHART
          </button>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '20px' }}>
          {/* WS indicator */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px',
            background: 'rgba(255, 255, 255, 0.02)',
            padding: '6px 14px',
            borderRadius: '20px',
            border: '1px solid rgba(255, 255, 255, 0.04)',
          }}>
            <div className={`dot ${wsConnected ? 'dot-online' : 'dot-error'}`} />
            <span style={{ fontFamily: 'Space Mono', fontSize: '9px', fontWeight: 700, color: wsConnected ? 'var(--buy)' : 'var(--sell)', letterSpacing: '0.5px' }}>
              {wsConnected ? 'SYSTEM ACTIVE' : 'CONNECTING'}
            </span>
          </div>

          {/* Mode badge */}
          {status && (
            <span className={`badge ${status.dry_run ? 'badge-dry' : 'badge-live'}`} style={{ padding: '6px 14px', fontSize: '9px' }}>
              {status.dry_run ? '🧪 DRY RUN' : '💼 INSTITUTIONAL LIVE'}
            </span>
          )}
        </div>
      </header>

      {/* Main content view */}
      <main style={{ padding: '24px 32px', maxWidth: '1680px', margin: '0 auto', width: '100%', flex: 1 }}>
        {activeTab === 'tradingview' ? (
          <TradingViewChart symbol={status?.pairs?.[0] || 'XAUUSD'} />
        ) : (
          <div className="dashboard-layout">
            {/* Left Column: Operation Center, Equity, Open Positions */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <StatusBar status={status} onPause={onPause} onResume={onResume} />
              <TradingController
                paused={status?.paused}
                currentPairs={status?.pairs}
                disableRiskGate={status?.disable_risk_gate}
                onToggleRiskGate={onToggleRiskGate}
                onPause={onPause}
                onResume={onResume}
                fetchAll={fetchAll}
                autoScalpMode={status?.auto_scalp_mode}
                onToggleAutoScalp={onToggleAutoScalp}
                autoScalpSL={status?.auto_scalp_sl}
                autoScalpTP={status?.auto_scalp_tp}
                autoScalpCycle={status?.auto_scalp_cycle}
              />
              <EquityCurve data={equity} />
              <OpenTrades trades={trades} />
            </div>

            {/* Right Column: P&L Summary, Lot Settings, Emergency, LLM Decision Log */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <PnlSummary pnl={pnl} status={status} />
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
                <LotSizeEditor currentLot={status?.lot_size} onUpdate={onLotUpdate} />
                <KillSwitch onKill={onKill} paused={status?.paused} onPause={onPause} onResume={onResume} />
              </div>

              <DecisionFeed decisions={decisions} />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
