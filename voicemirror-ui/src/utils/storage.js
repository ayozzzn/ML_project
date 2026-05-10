const KEY = 'mv_entries_v2'

export function loadEntries() {
  try { return JSON.parse(localStorage.getItem(KEY) || '[]') }
  catch { return [] }
}

export function saveEntry(entry) {
  const entries = loadEntries()
  entries.unshift({ ...entry, id: Date.now(), timestamp: Date.now() })
  if (entries.length > 200) entries.splice(200)
  localStorage.setItem(KEY, JSON.stringify(entries))
  return entries
}

export function clearEntries() { localStorage.removeItem(KEY) }

export const EMOTIONS = {
  happy:   { emoji: '😊', color: '#ffd84a', label: 'HAPPY',   desc: 'Joy & positivity' },
  calm:    { emoji: '😌', color: '#4dffa0', label: 'CALM',    desc: 'Relaxed & at ease' },
  neutral: { emoji: '😐', color: '#80b8ff', label: 'NEUTRAL', desc: 'Balanced state' },
  sad:     { emoji: '😢', color: '#5b7fff', label: 'SAD',     desc: 'Low mood' },
  angry:   { emoji: '😠', color: '#ff6060', label: 'ANGRY',   desc: 'Frustration' },
  anxiety: { emoji: '😰', color: '#c96cff', label: 'ANXIETY', desc: 'Stress & worry' },
}

export function computeWeeklyTrend(entries) {
  if (entries.length < 4) return null
  const now = Date.now()
  const week = 7 * 24 * 3600 * 1000
  const thisWeek = entries.filter(e => now - e.timestamp < week)
  const lastWeek = entries.filter(e => now - e.timestamp >= week && now - e.timestamp < week * 2)
  if (!thisWeek.length || !lastWeek.length) return null
  const avg = (arr, em) => arr.reduce((s, e) => s + (e.probs?.[em] || 0), 0) / arr.length
  return Object.fromEntries(
    Object.keys(EMOTIONS).map(em => {
      const curr = avg(thisWeek, em), prev = avg(lastWeek, em)
      const pct = prev > 0 ? ((curr - prev) / prev) * 100 : 0
      return [em, { curr: Math.round(curr * 100), prev: Math.round(prev * 100), pct: Math.round(pct) }]
    })
  )
}