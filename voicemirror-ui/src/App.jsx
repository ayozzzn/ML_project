import { useState } from 'react'
import Nav from './components/Nav.jsx'
import Background from './components/Background.jsx'
import RecordPage from './pages/RecordPage.jsx'
import TrendsPage from './pages/TrendsPage.jsx'
import AboutPage from './pages/AboutPage.jsx'
import './styles/global.css'

export default function App() {
  const [page, setPage] = useState('record')

  return (
    <div style={{ minHeight: '100vh', position: 'relative' }}>
      <Background />
      <Nav page={page} setPage={setPage} />
      <main style={{ maxWidth: 860, margin: '0 auto', padding: '40px 24px 80px' }}>
        {page === 'record' && <RecordPage />}
        {page === 'trends' && <TrendsPage />}
        {page === 'about'  && <AboutPage />}
      </main>
    </div>
  )
}