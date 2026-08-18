import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Activity } from 'lucide-react';

export default function EquityCurve({ data }) {
  // Format data for chart
  const chartData = data.map(d => ({
    time: new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    equity: d.equity,
    balance: d.balance
  }));

  return (
    <div className="card fade-in" style={{ flex: 1, minHeight: '340px' }}>
      <div className="card-header">
        <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={14} color="var(--accent)" />
          NEURAL EQUITY CURVE
        </div>
        <div className="card-tag">Growth Metric</div>
      </div>
      
      <div className="card-body" style={{ height: '280px', padding: '20px 20px 10px 0px' }}>
        {chartData.length === 0 ? (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontFamily: 'Space Mono', fontSize: '11px' }}>
            No equity snapshots logged yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.15}/>
                  <stop offset="95%" stopColor="var(--accent)" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.03)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="var(--text2)" 
                fontSize={9} 
                tickLine={false}
                axisLine={false}
                dy={10}
                style={{ fontFamily: 'Space Mono', opacity: 0.6 }}
              />
              <YAxis 
                stroke="var(--text2)" 
                fontSize={9} 
                tickLine={false}
                axisLine={false}
                domain={['auto', 'auto']}
                dx={-6}
                style={{ fontFamily: 'Space Mono', opacity: 0.6 }}
              />
              <Tooltip 
                contentStyle={{ 
                  background: 'var(--surface2)', 
                  borderColor: 'var(--border)', 
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--text)',
                  fontSize: '11px',
                  fontFamily: 'Space Mono',
                  boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)',
                  border: '1px solid var(--border)'
                }} 
              />
              <Area 
                type="monotone" 
                dataKey="equity" 
                stroke="var(--accent)" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorEquity)" 
                name="Equity"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
