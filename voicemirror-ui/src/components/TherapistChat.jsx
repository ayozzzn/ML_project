import { useState, useRef, useEffect, useCallback } from 'react'
import s from '../styles/TherapistChat.module.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EMOTION_COLORS = {
  happy:   '#ffd84a', calm: '#4dffa0', neutral: '#80b8ff',
  sad:     '#5b7fff', angry: '#ff6060', anxiety: '#c96cff',
}
const EMOTION_EMOJIS = {
  happy: '😊', calm: '😌', neutral: '😐', sad: '😢', angry: '😠', anxiety: '😰',
}
const EMOTION_RU = {
  happy: 'радость', calm: 'спокойствие', neutral: 'нейтральный',
  sad: 'грусть', angry: 'злость', anxiety: 'тревога',
}
const GREETINGS = {
  sad:     'Я здесь рядом. Расскажи мне — что случилось? Говори голосом или пиши — как удобнее.',
  anxiety: 'Слышу напряжение в твоём голосе. Ты в безопасности прямо сейчас. Что тебя беспокоит?',
  angry:   'Слышу раздражение — оно имеет право быть. Что именно задело тебя сильнее всего?',
  happy:   'В твоём голосе радость! Поделись — что хорошего произошло?',
  calm:    'Ты сейчас в хорошем месте. Расскажи мне о своём дне.',
  neutral: 'Привет. Как прошёл твой день? Говори или пиши — я здесь.',
}

function EmotionPill({ emotion, confidence }) {
  const color = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral
  return (
    <span className={s.emotionPill} style={{ color, borderColor: color + '40', background: color + '12' }}>
      {EMOTION_EMOJIS[emotion]} {EMOTION_RU[emotion]}{confidence ? ` · ${confidence}%` : ''}
    </span>
  )
}

export default function TherapistChat({ emotion: initialEmotion }) {
  const [messages, setMessages]               = useState([])
  const [input, setInput]                     = useState('')
  const [isTyping, setIsTyping]               = useState(false)
  const [currentEmotion, setCurrentEmotion]   = useState(initialEmotion || 'neutral')
  const [isRecording, setIsRecording]         = useState(false)
  const [recSecs, setRecSecs]                 = useState(0)
  const [micError, setMicError]               = useState(null)
  const [whisperAvailable, setWhisperAvailable] = useState(null)
  const [sessionStats, setSessionStats]       = useState(null)

  const messagesEndRef = useRef(null)
  const mediaRef       = useRef(null)
  const chunksRef      = useRef([])
  const timerRef       = useRef(null)
  const inputRef       = useRef(null)

  const color = EMOTION_COLORS[currentEmotion] || EMOTION_COLORS.neutral

  useEffect(() => {
    setTimeout(() => {
      setMessages([{ id: 1, role: 'assistant', content: GREETINGS[initialEmotion] || GREETINGS.neutral, emotion: initialEmotion }])
    }, 300)
    fetch(`${API_BASE}/health`).then(r => r.json())
      .then(d => setWhisperAvailable(d.whisper_available ?? false))
      .catch(() => setWhisperAvailable(false))
  }, [initialEmotion])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // ── Text ─────────────────────────────────────────────────────────────────────
  const sendText = useCallback(async () => {
    const text = input.trim()
    if (!text || isTyping) return
    setInput('')
    const uid = Date.now()
    setMessages(prev => [...prev, { id: uid, role: 'user', content: text, emotion: currentEmotion }])
    setIsTyping(true)
    try {
      const history = messages.map(m => ({ role: m.role, content: m.content || '' }))
      const res  = await fetch(`${API_BASE}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion: currentEmotion, message: text, history }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        id: Date.now(), role: 'assistant', emotion: currentEmotion,
        content: data.response || 'Прости, не получилось ответить.',
        quote: data.quote, grounding: data.grounding,
      }])
    } catch {
      setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', emotion: currentEmotion, content: _fallback(currentEmotion) }])
    } finally {
      setIsTyping(false)
      inputRef.current?.focus()
    }
  }, [input, isTyping, currentEmotion, messages])

  // ── Voice ─────────────────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    setMicError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []
      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        processVoice(new File([blob], 'voice.webm', { type: 'audio/webm' }))
      }
      recorder.onerror = () => { setMicError('Ошибка записи'); setIsRecording(false) }
      mediaRef.current = recorder
      recorder.start(1000)
      setIsRecording(true); setRecSecs(0)
      timerRef.current = setInterval(() => setRecSecs(p => p + 1), 1000)
    } catch {
      setMicError('Нет доступа к микрофону')
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRef.current?.state === 'recording') mediaRef.current.stop()
    clearInterval(timerRef.current)
    setIsRecording(false); setRecSecs(0)
  }, [])

  // ── Full pipeline: emotion + Whisper + LLM ───────────────────────────────────
  const processVoice = useCallback(async (file) => {
    const pid = Date.now()
    setMessages(prev => [...prev, { id: pid, role: 'user', pending: true, emotion: currentEmotion }])
    setIsTyping(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('language', 'ru')

      const res  = await fetch(`${API_BASE}/voice-chat`, { method: 'POST', body: form })
      const data = await res.json()
      const emo  = data.emotion || currentEmotion

      // Заменяем плейсхолдер реальным сообщением
      setMessages(prev => prev.map(m => m.id === pid ? {
        ...m, pending: false,
        content: data.user_text?.trim() || null,  // текст от Whisper
        voiceLabel: true,
        detectedEmotion: emo,
        confidence: data.confidence,
        whisperAvailable: data.whisper_available,
      } : m))

      setMessages(prev => [...prev, {
        id: Date.now(), role: 'assistant', emotion: emo,
        content: data.therapist_response || 'Я слышу тебя.',
        quote: data.quote, grounding: data.grounding,
      }])

      setCurrentEmotion(emo)
      setWhisperAvailable(data.whisper_available)

      if (data.conversation_length > 1) {
        fetch(`${API_BASE}/voice-chat/summary`).then(r => r.json()).then(setSessionStats).catch(() => {})
      }
    } catch {
      setMessages(prev => prev.map(m => m.id === pid
        ? { ...m, pending: false, voiceLabel: true, detectedEmotion: currentEmotion }
        : m
      ))
      setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', emotion: currentEmotion, content: _fallback(currentEmotion) }])
    } finally {
      setIsTyping(false)
    }
  }, [currentEmotion])

  const fmt = n => `${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`

  const resetSession = async () => {
    await fetch(`${API_BASE}/voice-chat/reset`, { method: 'POST' }).catch(() => {})
    setMessages([{ id: Date.now(), role: 'assistant', content: GREETINGS[currentEmotion] || GREETINGS.neutral, emotion: currentEmotion }])
    setSessionStats(null)
  }

  return (
    <div className={s.root} style={{ '--chat-accent': color }}>

      {/* Header */}
      <div className={s.header}>
        <div className={s.headerLeft}>
          <div className={s.headerDot} style={{ background: color, boxShadow: `0 0 10px ${color}80` }} />
          <div>
            <div className={s.headerTitle}>MindVoice Psychologist</div>
            <div className={s.headerSub}>
              <span style={{ color }}>{EMOTION_EMOJIS[currentEmotion]} {EMOTION_RU[currentEmotion]}</span>
              {whisperAvailable === true  && <span className={s.whisperOn}>✦ Whisper ASR активен</span>}
              {whisperAvailable === false && <span className={s.whisperOff}>только эмоция</span>}
            </div>
          </div>
        </div>
        <div className={s.headerRight}>
          {sessionStats?.has_history && (
            <span className={s.sessionInfo}>
              {sessionStats.total_messages} сообщ. · {EMOTION_EMOJIS[sessionStats.dominant_emotion]} доминирует
            </span>
          )}
          <button className={s.resetIconBtn} onClick={resetSession} title="Новый разговор">↺</button>
        </div>
      </div>

      {/* Whisper hint */}
      {whisperAvailable === false && (
        <div className={s.whisperHint}>
          ℹ Whisper не установлен — психолог слышит эмоцию, но не слова.
          Для полного пайплайна: <code>pip install faster-whisper</code>
        </div>
      )}

      {/* Messages */}
      <div className={s.messages}>
        {messages.map(msg => (
          <div key={msg.id} className={`${s.msg} ${msg.role === 'user' ? s.msgUser : s.msgAssistant}`}>

            {msg.role === 'assistant' && (
              <div className={s.avatar} style={{ borderColor: EMOTION_COLORS[msg.emotion] || color }}>
                <span style={{ color: EMOTION_COLORS[msg.emotion] || color, fontSize: 16 }}>◈</span>
              </div>
            )}

            <div className={s.msgBody}>
              {/* Голосовое — строка с эмоцией */}
              {msg.voiceLabel && (
                <div className={s.voiceRow}>
                  <span className={s.voiceIcon}>🎙</span>
                  <span className={s.voiceLbl}>Голосовое</span>
                  {msg.detectedEmotion && <EmotionPill emotion={msg.detectedEmotion} confidence={msg.confidence} />}
                  {msg.whisperAvailable === false && <span className={s.noTranscript}>без транскрипта</span>}
                </div>
              )}

              {/* Основное тело */}
              {msg.pending ? (
                <div className={s.bubble} style={{ borderColor: color + '30' }}>
                  <div className={s.pendingDots}>
                    <span style={{ background: color }} /><span style={{ background: color }} /><span style={{ background: color }} />
                  </div>
                </div>
              ) : msg.content ? (
                <div className={s.bubble} style={msg.role === 'user' ? {
                  background: `${EMOTION_COLORS[msg.detectedEmotion || msg.emotion] || color}0e`,
                  borderColor: `${EMOTION_COLORS[msg.detectedEmotion || msg.emotion] || color}28`,
                } : {}}>
                  {msg.content}
                </div>
              ) : null}

              {/* Цитата */}
              {msg.quote && (
                <div className={s.quoteBlock} style={{ borderColor: (EMOTION_COLORS[msg.emotion] || color) + '35' }}>
                  <span className={s.qMark} style={{ color: EMOTION_COLORS[msg.emotion] || color }}>"</span>
                  {msg.quote.replace(/[✨🌧️🕯️🌊🌬️💨🤲🌟☀️🧘🌿🔥]/g, '').trim()}
                </div>
              )}

              {/* Практика */}
              {msg.grounding && (
                <div className={s.groundingBlock} style={{
                  borderColor: (EMOTION_COLORS[msg.emotion] || color) + '35',
                  background:  (EMOTION_COLORS[msg.emotion] || color) + '09',
                }}>
                  <div className={s.groundingLbl} style={{ color: EMOTION_COLORS[msg.emotion] || color }}>
                    ◎ Попробуй прямо сейчас
                  </div>
                  <div className={s.groundingTxt}>{msg.grounding.replace(/[🌬️💨🤲🎯🌿]/g, '').trim()}</div>
                </div>
              )}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className={`${s.msg} ${s.msgAssistant}`}>
            <div className={s.avatar} style={{ borderColor: color }}>
              <span style={{ color, fontSize: 16 }}>◈</span>
            </div>
            <div className={s.msgBody}>
              <div className={s.bubble}>
                <div className={s.typingDots}>
                  <span style={{ background: color }} /><span style={{ background: color }} /><span style={{ background: color }} />
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className={s.inputArea}>
        {micError && <div className={s.micError}>{micError}</div>}

        {isRecording && (
          <div className={s.recBar}>
            <span className={s.recDot} />
            <span className={s.recTime}>{fmt(recSecs)}</span>
            <div className={s.recWave}>
              {Array.from({ length: 16 }).map((_, i) => (
                <span key={i} className={s.recBar2} style={{ background: color, animationDelay: `${i * 0.07}s` }} />
              ))}
            </div>
            <span className={s.recHint}>Отпусти чтобы отправить</span>
          </div>
        )}

        <div className={s.inputRow}>
          <button
            className={`${s.voiceBtn} ${isRecording ? s.voiceBtnRec : ''}`}
            style={{ '--c': color }}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onTouchStart={e => { e.preventDefault(); startRecording() }}
            onTouchEnd={e => { e.preventDefault(); stopRecording() }}
            disabled={isTyping}
            title="Зажми и говори"
          >
            {isRecording ? '⏹' : '🎙'}
          </button>

          <textarea
            ref={inputRef}
            className={s.input}
            style={{ '--fc': color + '55' }}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText() } }}
            placeholder={isRecording ? 'Запись...' : 'Напиши что-нибудь... (Enter — отправить)'}
            rows={2}
            disabled={isTyping || isRecording}
          />

          <button
            className={s.sendBtn}
            style={{ '--c': color }}
            onClick={sendText}
            disabled={!input.trim() || isTyping || isRecording}
          >↑</button>
        </div>

        <div className={s.inputHint}>
          {whisperAvailable
            ? '🎙 Голос → Whisper транскрибирует текст → психолог читает и слова и эмоцию'
            : '🎙 Зажми микрофон и говори  ·  или напиши текстом  ·  Enter отправляет'}
        </div>
      </div>
    </div>
  )
}

function _fallback(e) {
  return ({
    sad: 'Я слышу тебя. Это звучит тяжело — расскажи больше, если хочешь.',
    anxiety: 'Твои чувства реальны. Давай замедлимся — сделай один глубокий вдох прямо сейчас.',
    angry: 'Что именно задело тебя сильнее всего? Я здесь и не буду осуждать.',
    happy: 'Это замечательно! Что именно сделало тебя счастливее сегодня?',
    calm: 'Как ты пришёл к этому спокойствию? Расскажи.',
    neutral: 'Как ты себя чувствуешь прямо сейчас — в теле, в мыслях?',
  })[e] || 'Я здесь. Расскажи мне.'
}