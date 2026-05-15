import { useState } from 'react'
import Nav from './components/Nav.jsx'
import ParticleBackground from './components/ParticleBackground.jsx'
import RecordPage from './pages/RecordPage.jsx'
import TrendsPage from './pages/TrendsPage.jsx'
import AboutPage from './pages/AboutPage.jsx'
import './styles/global.css'

export default function App() {
  const [page, setPage] = useState('record')

  return (
    <div style={{ minHeight: '100vh', position: 'relative' }}>
      <ParticleBackground />

      {/* Subtle radial glow at top */}
      <div style={{
        position: 'fixed',
        top: -200,
        left: '50%',
        transform: 'translateX(-50%)',
        width: 800,
        height: 600,
        background: 'radial-gradient(ellipse, rgba(116,185,255,0.06) 0%, transparent 70%)',
        pointerEvents: 'none',
        zIndex: 0,
      }} />

      <Nav page={page} setPage={setPage} />

      <main style={{
        maxWidth: 880,
        margin: '0 auto',
        padding: '40px 24px 100px',
        position: 'relative',
        zIndex: 1,
      }}>
        {page === 'record' && <RecordPage />}
        {page === 'trends' && <TrendsPage />}
        {page === 'about'  && <AboutPage />}
      </main>
    </div>
  )
}