import s from '../styles/Nav.module.css'

const tabs = [
  { id: 'record', label: 'RECORD' },
  { id: 'trends', label: 'TRENDS' },
  { id: 'about',  label: 'ABOUT'  },
]

export default function Nav({ page, setPage }) {
  return (
    <nav className={s.nav}>
      <div className={s.logo}>
        <span className={s.logoIcon}>◈</span>
        <span className={s.logoText}>VOICE<span className={s.logoAccent}>MIRROR</span></span>
      </div>

      <div className={s.tabs}>
        {tabs.map(t => (
          <button
            key={t.id}
            className={`${s.tab} ${page === t.id ? s.active : ''}`}
            onClick={() => setPage(t.id)}
          >
            {t.label}
            {page === t.id && <span className={s.tabDot} />}
          </button>
        ))}
      </div>

      <div className={s.status}>
        <span className={s.statusDot} />
        <span className={s.statusText}>LIVE</span>
      </div>
    </nav>
  )
}