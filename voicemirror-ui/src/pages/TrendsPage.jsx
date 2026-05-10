import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, Legend
} from 'recharts'
import { loadEntries, computeWeeklyTrend, EMOTIONS } from '../utils/storage.js'
import s from '../styles/TrendsPage.module.css'

const EMO_KEYS = Object.keys(EMOTIONS)
const EM_COLORS = {
  happy:'#ffd84a', calm:'#4dffa0', neutral:'#80b8ff',
  sad:'#5b7fff', angry:'#ff6060', anxiety:'#c96cff'
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className={s.tooltip}>
      <div className={s.tooltipDate}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} className={s.tooltipRow}>
          <span style={{ color: p.color }}>■</span>
          <span>{p.dataKey}</span>
          <span>{Math.round(p.value)}%</span>
        </div>
      ))}
    </div>
  )
}

export default function TrendsPage() {
  const [entries, setEntries] = useState([])
  const [view, setView]       = useState('line')
  const [range, setRange]     = useState(14) 

  useEffect(() => { setEntries(loadEntries()) }, [])

  const cutoff = Date.now() - range * 24 * 3600 * 1000
  const recent = entries.filter(e => e.timestamp >= cutoff)

  const byDate = {}
  recent.forEach(e => {
    const d = new Date(e.timestamp).toLocaleDateString('en-US', { month:'short', day:'numeric' })
    if (!byDate[d]) byDate[d] = { date: d, count: 0, ...Object.fromEntries(EMO_KEYS.map(k => [k, 0])) }
    EMO_KEYS.forEach(k => { byDate[d][k] += (e.probs?.[k] || 0) })
    byDate[d].count++
  })
  const chartData = Object.values(byDate).map(row => {
    const out = { date: row.date }
    EMO_KEYS.forEach(k => { out[k] = row.count ? Math.round((row[k] / row.count) * 100) : 0 })
    return out
  })

  const emotionCounts = Object.fromEntries(EMO_KEYS.map(k => [k, 0]))
  recent.forEach(e => { if (e.emotion) emotionCounts[e.emotion]++ })
  const pieData = EMO_KEYS.map(k => ({ name: k, value: emotionCounts[k] })).filter(d => d.value > 0)

  const trend = computeWeeklyTrend(entries)

  const total = entries.length
  const dom = entries.length
    ? EMO_KEYS.reduce((a,b) => emotionCounts[a] >= emotionCounts[b] ? a : b)
    : null
  const avgConf = entries.length
    ? Math.round(entries.reduce((s,e) => s + (e.confidence||0), 0) / entries.length)
    : 0
  const anxietyPct = entries.length
    ? Math.round((emotionCounts.anxiety / entries.length) * 100)
    : 0

  const empty = recent.length === 0

  return (
    <div className={s.root}>
      <div className={s.hero}>
        <div className={s.heroLabel}>EMOTIONAL PATTERNS</div>
        <h1 className={s.heroTitle}>Your <span className={s.heroAccent}>Journey</span></h1>
        <p className={s.heroSub}>Track how your emotional state shifts over time</p>
      </div>

      <div className={s.statsGrid}>
        {[
          { label: 'TOTAL ENTRIES', value: total, sub: 'all time', color: 'var(--c7)' },
          { label: 'DOMINANT MOOD', value: dom ? EMOTIONS[dom]?.emoji + ' ' + dom?.toUpperCase() : '—', sub: `last ${range} days`, color: dom ? EM_COLORS[dom] : 'var(--text-muted)' },
          { label: 'AVG CONFIDENCE', value: `${avgConf}%`, sub: 'model certainty', color: 'var(--c8)' },
          { label: 'ANXIETY RATE', value: `${anxietyPct}%`, sub: anxietyPct > 30 ? '▲ elevated' : '▼ normal', color: anxietyPct > 30 ? '#ff6060' : '#4dffa0' },
        ].map((stat, i) => (
          <div key={i} className={s.statCard} style={{ '--stat-color': stat.color }}>
            <div className={s.statLabel}>{stat.label}</div>
            <div className={s.statValue}>{stat.value}</div>
            <div className={s.statSub}>{stat.sub}</div>
          </div>
        ))}
      </div>

      {trend && (
        <div className={s.insightCard}>
          <div className={s.insightTitle}>◈ WEEKLY INSIGHT</div>
          <div className={s.insightGrid}>
            {Object.entries(trend).map(([em, data]) => (
              <div key={em} className={s.insightRow}>
                <span className={s.insightEmoji}>{EMOTIONS[em]?.emoji}</span>
                <span className={s.insightName}>{em}</span>
                <div className={s.insightBar}>
                  <div className={s.insightFill} style={{ width: `${data.curr}%`, background: EM_COLORS[em] }} />
                </div>
                <span className={s.insightPct} style={{ color: data.pct > 0 ? '#ff6060' : data.pct < 0 ? '#4dffa0' : 'var(--text-muted)' }}>
                  {data.pct > 0 ? '+' : ''}{data.pct}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={s.chartCard}>
        <div className={s.chartHeader}>
          <div className={s.chartTitle}>EMOTION TREND</div>
          <div className={s.controls}>
            <div className={s.ctrlGroup}>
              {[7,14,30].map(d => (
                <button key={d} className={`${s.ctrlBtn} ${range===d?s.ctrlActive:''}`} onClick={() => setRange(d)}>
                  {d}D
                </button>
              ))}
            </div>
            <div className={s.ctrlGroup}>
              {['line','area','pie'].map(v => (
                <button key={v} className={`${s.ctrlBtn} ${view===v?s.ctrlActive:''}`} onClick={() => setView(v)}>
                  {v.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {empty ? (
          <div className={s.emptyChart}>
            <div className={s.emptyIcon}>◈</div>
            <div className={s.emptyTxt}>No data for this period</div>
            <div className={s.emptySub}>Record a voice diary to start tracking</div>
          </div>
        ) : (
          <div className={s.chartArea}>
            {view === 'line' && (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData} margin={{ top:5, right:5, bottom:5, left:-20 }}>
                  <XAxis dataKey="date" tick={{ fill:'rgba(170,197,255,0.4)', fontSize:11, fontFamily:'Josefin Sans' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'rgba(170,197,255,0.3)', fontSize:10, fontFamily:'Josefin Sans' }} axisLine={false} tickLine={false} domain={[0,100]} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  {EMO_KEYS.map(k => (
                    <Line key={k} type="monotone" dataKey={k} stroke={EM_COLORS[k]} strokeWidth={2}
                      dot={{ fill: EM_COLORS[k], r: 3, strokeWidth: 0 }}
                      activeDot={{ r: 5, strokeWidth: 0 }} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
            {view === 'area' && (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartData} margin={{ top:5, right:5, bottom:5, left:-20 }}>
                  <XAxis dataKey="date" tick={{ fill:'rgba(170,197,255,0.4)', fontSize:11, fontFamily:'Josefin Sans' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill:'rgba(170,197,255,0.3)', fontSize:10, fontFamily:'Josefin Sans' }} axisLine={false} tickLine={false} domain={[0,100]} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  {EMO_KEYS.map(k => (
                    <Area key={k} type="monotone" dataKey={k} stroke={EM_COLORS[k]} strokeWidth={1.5}
                      fill={EM_COLORS[k]} fillOpacity={0.07} stackId={null} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            )}
            {view === 'pie' && (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} innerRadius={50}
                    dataKey="value" paddingAngle={3}
                    label={({ name, percent }) => `${name} ${Math.round(percent*100)}%`}
                    labelLine={{ stroke: 'rgba(170,197,255,0.3)' }}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={EM_COLORS[entry.name]} fillOpacity={0.8} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => [v, 'entries']} contentStyle={{ background:'var(--bg-card)', border:'1px solid var(--border)', borderRadius:12, fontFamily:'Josefin Sans' }} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        )}
      </div>

      <div className={s.entriesCard}>
        <div className={s.entriesTitle}>RECENT ENTRIES</div>
        {entries.length === 0 ? (
          <div className={s.emptyEntries}>No entries yet — record your first voice diary</div>
        ) : (
          <div className={s.entriesList}>
            {entries.slice(0, 10).map((e, i) => {
              const em = EMOTIONS[e.emotion] || {}
              const d = new Date(e.timestamp)
              const dateStr = d.toLocaleDateString('en-US', { month:'short', day:'numeric' }) +
                              ' · ' + d.toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit' })
              return (
                <div key={e.id || i} className={s.entryRow}>
                  <div className={s.entryEmoji}>{em.emoji || '❓'}</div>
                  <div className={s.entryInfo}>
                    <div className={s.entryEmotion} style={{ color: EM_COLORS[e.emotion] }}>
                      {em.label || e.emotion?.toUpperCase()}
                    </div>
                    <div className={s.entryDate}>{dateStr}</div>
                  </div>
                  <div className={s.entryConf}>{e.confidence}%</div>
                  <div className={s.entryBar}>
                    {Object.entries(e.probs || {}).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,v]) => (
                      <div key={k} className={s.entryMini} style={{ width:`${Math.round(v*100)}%`, background: EM_COLORS[k] }} />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}