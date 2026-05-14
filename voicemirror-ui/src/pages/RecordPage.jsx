import { useState, useRef, useCallback } from 'react'
import Waveform from '../components/Waveform.jsx'
import EmotionResult from '../components/EmotionResult.jsx'
import TherapistChat from '../components/TherapistChat.jsx'
import { saveEntry } from '../utils/storage.js'
import { uploadAudio } from '../utils/api.js'
import s from '../styles/RecordPage.module.css'

const EMOTION_LABELS_RU = {
  happy:   'Радость',
  calm:    'Спокойствие',
  neutral: 'Нейтральный',
  sad:     'Грусть',
  angry:   'Злость',
  anxiety: 'Тревога',
}

export default function RecordPage() {
  const [state, setState]           = useState('idle')
  const [result, setResult]         = useState(null)
  const [fileName, setFileName]     = useState(null)
  const [fileObj, setFileObj]       = useState(null)
  const [recSecs, setRecSecs]       = useState(0)
  const [note, setNote]             = useState('')
  const [apiError, setApiError]     = useState(null)
  const [isDemo, setIsDemo]         = useState(false)
  const [micError, setMicError]     = useState(null)
  const [showTherapist, setShowTherapist] = useState(false)

  const timerRef  = useRef(null)
  const fileRef   = useRef(null)
  const mediaRef  = useRef(null)
  const chunksRef = useRef([])

  const isActive   = state === 'recording'
  const canAnalyze = state === 'uploading' && fileObj !== null

  const toggleRecord = useCallback(async () => {
    if (state === 'recording') {
      if (mediaRef.current?.state === 'recording') mediaRef.current.stop()
      clearInterval(timerRef.current)
    } else if (state === 'idle') {
      setMicError(null)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
        chunksRef.current = []

        recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
          const file = new File([blob], 'recording.webm', { type: 'audio/webm' })
          setFileObj(file)
          setFileName('recording.webm')
          setState('uploading')
          stream.getTracks().forEach(t => t.stop())
        }
        recorder.onerror = () => { setMicError('Ошибка записи'); setState('idle') }

        mediaRef.current = recorder
        recorder.start(1000)
        setRecSecs(0); setResult(null); setApiError(null)
        setState('recording')
        timerRef.current = setInterval(() => setRecSecs(p => p + 1), 1000)
      } catch {
        setMicError('Нет доступа к микрофону. Проверьте разрешения.')
      }
    }
  }, [state])

  const handleFile = useCallback(e => {
    const f = e.target.files?.[0]
    if (!f) return
    setFileName(f.name); setFileObj(f); setResult(null); setApiError(null); setMicError(null)
    setState('uploading')
  }, [])

  const handleDrop = useCallback(e => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (!f) return
    setFileName(f.name); setFileObj(f); setResult(null); setApiError(null); setMicError(null)
    setState('uploading')
  }, [])

  const analyze = useCallback(async () => {
    if (!fileObj) return
    setState('analyzing'); setApiError(null)
    const res = await uploadAudio(fileObj, note)
    if (res.error) { setApiError(res.error); setState('uploading'); return }
    setResult(res)
    setIsDemo(res.demo || false)
    saveEntry({ emotion: res.emotion, probs: res.probabilities, confidence: res.confidence, advice: res.advice })
    setState('done')
  }, [fileObj, note])

  const reset = () => {
    if (mediaRef.current?.state === 'recording') mediaRef.current.stop()
    clearInterval(timerRef.current)
    setState('idle'); setResult(null); setFileName(null); setFileObj(null)
    setNote(''); setRecSecs(0); setApiError(null); setMicError(null); setShowTherapist(false)
  }

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  return (
    <div className={s.root}>
      <div className={s.hero}>
        <div className={s.heroLabel}>VOICE EMOTION ANALYSIS</div>
        <h1 className={s.heroTitle}>How are you<br /><span className={s.heroAccent}>feeling today?</span></h1>
        <p className={s.heroSub}>
          Speak freely for 10–60 seconds. MindVoice analyzes your pitch, energy, and pauses to detect your emotional state.
        </p>
      </div>

      <div className={s.recordCard}>
        <Waveform active={isActive} />

        <div className={s.btnArea}>
          {state !== 'analyzing' && state !== 'done' && (
            <>
              <div className={s.recordOuter}>
                {isActive && (<><div className={`${s.ring}`} /><div className={`${s.ring} ${s.ring2}`} /></>)}
                <button
                  className={`${s.recordBtn} ${isActive ? s.recording : ''}`}
                  onClick={toggleRecord}
                  aria-label={isActive ? 'Остановить' : 'Начать запись'}
                  disabled={!!micError}
                >
                  {isActive ? '⏹' : '🎙'}
                </button>
              </div>
              <div className={s.statusRow}>
                {state === 'idle' && !micError && <span className={s.statusTxt}>TAP TO START RECORDING</span>}
                {state === 'idle' && micError  && <span className={s.statusError}>{micError}</span>}
                {state === 'recording'         && <span className={s.statusRec}><span className={s.recDot} /> RECORDING {fmt(recSecs)}</span>}
                {state === 'uploading'         && <span className={s.statusDone}>✓ READY — {fileName}</span>}
              </div>
            </>
          )}

          {state === 'analyzing' && (
            <div className={s.analyzing}>
              <div className={s.analyzeSpinner} />
              <div className={s.analyzeTxt}>ANALYZING YOUR VOICE</div>
              <div className={s.analyzeSteps}>
                {['Noise reduction', 'Feature extraction', 'SHAP inference'].map((step, i) => (
                  <div key={i} className={s.analyzeStep} style={{ animationDelay: `${i * 0.7}s` }}>{step}</div>
                ))}
              </div>
            </div>
          )}

          {state === 'done' && (
            <button className={s.resetBtn} onClick={reset}>↺ ANALYZE AGAIN</button>
          )}
        </div>

        {apiError && <div className={s.errorBanner}>⚠ {apiError}</div>}

        {(state === 'idle' || state === 'uploading') && (
          <div
            className={s.uploadZone}
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept=".wav,.mp3,.ogg,.flac,.m4a,.webm" onChange={handleFile} style={{ display: 'none' }} />
            <div className={s.uploadIcon}>↑</div>
            <div className={s.uploadTxt}>{fileName ? `✓ ${fileName}` : 'DRAG & DROP or click to upload'}</div>
            <div className={s.uploadSub}>WAV · MP3 · OGG · FLAC · M4A · WEBM</div>
          </div>
        )}

        {(state === 'uploading' || state === 'done') && (
          <textarea
            className={s.noteArea}
            placeholder="What's on your mind? Share your feelings..."
            value={note}
            onChange={e => setNote(e.target.value)}
            rows={2}
          />
        )}

        {isDemo && state === 'done' && (
          <div className={s.demoBanner}>
            ℹ Demo mode — FastAPI not reachable. For real predictions run:<br />
            <code>uvicorn src.api:app --reload --port 8000</code>
          </div>
        )}

        {canAnalyze && (
          <button className={s.analyzeBtn} onClick={analyze}>
            ANALYZE EMOTION <span className={s.analyzeBtnArrow}>→</span>
          </button>
        )}
      </div>

      {result && (
        <>
          {/* Low confidence warning */}
          {result.low_confidence && (
            <div className={s.errorBanner} style={{ background: 'rgba(255,216,74,0.1)', borderColor: 'rgba(255,216,74,0.3)', color: '#ffd84a' }}>
              ⚠ Model confidence is low ({result.confidence}%). Try recording a longer message for better accuracy.
            </div>
          )}

          <EmotionResult result={result} />

          {result.advice && (
            <div className={s.adviceCard}>
              <div className={s.adviceHeader}>
                <span className={s.adviceIcon}>🧠</span>
                <span className={s.adviceTitle}>MindVoice Therapist says:</span>
              </div>
              <div className={s.adviceText}>{result.advice}</div>
              {result.quote && <div className={s.adviceQuote}>✨ {result.quote}</div>}
              {result.grounding && (
                <div className={s.groundingBox}>
                  <div className={s.groundingTitle}>🌿 Try this grounding technique:</div>
                  <div className={s.groundingText}>{result.grounding}</div>
                </div>
              )}
            </div>
          )}

          <button className={s.chatBtn} onClick={() => setShowTherapist(true)}>
            🧠 Chat with AI Psychologist
            <span className={s.chatBtnArrow}>→</span>
          </button>
        </>
      )}

      {showTherapist && (
        <div className={s.modalOverlay} onClick={() => setShowTherapist(false)}>
          <div className={s.modal} onClick={e => e.stopPropagation()}>
            <TherapistChat
              emotion={result?.emotion || 'neutral'}
              onClose={() => setShowTherapist(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}