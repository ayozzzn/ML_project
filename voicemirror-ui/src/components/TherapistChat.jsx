import { useState, useRef, useEffect } from 'react'
import s from '../styles/TherapistChat.module.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function TherapistChat({ emotion, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [quote, setQuote] = useState(null)
  const [grounding, setGrounding] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const messagesEndRef = useRef(null)

  const quotesLibrary = [
    { text: "Всё, что происходит, происходит в нужное время.", author: "Лао-цзы", tags: ["спокойствие", "принятие"] },
    { text: "Ты не обязан быть совершенным. Ты обязан быть собой.", author: "Будда", tags: ["самопринятие", "спокойствие"] },
    { text: "И это пройдёт.", author: "Соломон", tags: ["надежда", "утешение"] },
    { text: "Тревога — это волна. Она приходит и уходит. Ты — океан.", author: "", tags: ["тревога", "спокойствие"] },
    { text: "Счастье можно найти даже в самые тёмные времена, если не забывать зажигать свет.", author: "Альбус Дамблдор", tags: ["надежда", "счастье"] },
    { text: "Ты гораздо сильнее, чем думаешь.", author: "", tags: ["уверенность", "поддержка"] },
    { text: "Позволь себе чувствовать. Эмоции — не слабость, а сигналы.", author: "", tags: ["принятие", "самопознание"] },
    { text: "Каждый день — это новая возможность начать иначе.", author: "", tags: ["надежда", "мотивация"] },
  ]

  // Initial greeting based on emotion
  useEffect(() => {
    const greetings = {
      sad: "Привет. Я здесь, и я вижу, что тебе сейчас непросто. Расскажи — что произошло? Я слушаю. 💙",
      anxiety: "Привет. Я чувствую, что ты сейчас в напряжении. Ты в безопасности, я рядом. Что тебя беспокоит? 🌊",
      angry: "Привет. Что-то явно задело тебя сегодня. Хочешь выговориться? Я не буду осуждать. 🔥",
      happy: "Привет! Чувствую твою энергию — что-то хорошее случилось? Расскажи! ✨",
      calm: "Привет. Ты сейчас в хорошем состоянии — это ценно. Как ты этого достиг? 🧘",
      neutral: "Привет! Я здесь, чтобы поговорить. Как прошёл твой день? 🌿",
    }
    setTimeout(() => {
      setMessages([{
        role: 'assistant',
        content: greetings[emotion] || greetings.neutral
      }])
    }, 400)
  }, [emotion])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // REST-based send (fixes broken WebSocket)
  const sendMessage = async () => {
    if (!input.trim() || isTyping) return

    const userMsg = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setIsTyping(true)

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }))

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion,
          message: userMsg,
          history,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response || 'Прости, не получилось ответить. Попробуй ещё раз.'
      }])

      if (data.quote) setQuote(data.quote)
      if (data.grounding) setGrounding(data.grounding)

    } catch (err) {
      console.error('Chat error:', err)
      // Graceful fallback
      const fallbacks = {
        sad: "Я слышу тебя. Это звучит тяжело. Ты не один в этом — расскажи больше, если хочешь. 💙",
        anxiety: "Твои чувства реальны и важны. Давай попробуем замедлиться вместе. Сделай глубокий вдох. 🌊",
        angry: "Это звучит очень frustrating. Что именно тебя задело больше всего? 🔥",
        happy: "Это замечательно! Радость заслуживает того, чтобы её разделить. ✨",
        calm: "Ты молодец, что заботишься о себе. Продолжай в том же духе. 🧘",
        neutral: "Спасибо, что поделился. Как ты себя чувствуешь прямо сейчас? 🌿",
      }
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: fallbacks[emotion] || fallbacks.neutral
      }])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const filteredQuotes = quotesLibrary.filter(q =>
    q.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
    q.tags.some(tag => tag.includes(searchTerm.toLowerCase()))
  )

  return (
    <div className={s.root}>
      <div className={s.header}>
        <div className={s.headerTitle}>
          <span className={s.headerIcon}>🧠</span>
          MindVoice Therapist
        </div>
        <button className={s.closeBtn} onClick={onClose} aria-label="Закрыть">✕</button>
      </div>

      <div className={s.mainGrid}>
        {/* Chat area */}
        <div className={s.chatArea}>
          <div className={s.messages}>
            {messages.map((msg, i) => (
              <div key={i} className={`${s.message} ${msg.role === 'user' ? s.userMsg : s.assistantMsg}`}>
                <div className={s.messageBubble}>
                  {msg.role === 'assistant' && <span className={s.avatar}>🧠</span>}
                  <div className={s.messageContent}>{msg.content}</div>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className={`${s.message} ${s.assistantMsg}`}>
                <div className={s.messageBubble}>
                  <span className={s.avatar}>🧠</span>
                  <div className={s.typing}>
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={s.inputArea}>
            <textarea
              className={s.input}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Write what's on your mind... (Enter to send)"
              rows={2}
              disabled={isTyping}
            />
            <button
              className={s.sendBtn}
              onClick={sendMessage}
              disabled={!input.trim() || isTyping}
            >
              {isTyping ? '...' : 'Send →'}
            </button>
          </div>
        </div>

        {/* Resources panel */}
        <div className={s.resourcesArea}>
          {quote && (
            <div className={s.resourceCard}>
              <div className={s.resourceTitle}>✨ Quote of the day</div>
              <div className={s.quoteText}>{quote}</div>
            </div>
          )}

          {grounding && (
            <div className={s.resourceCard}>
              <div className={s.resourceTitle}>🌿 Grounding technique</div>
              <div className={s.groundingText}>{grounding}</div>
            </div>
          )}

          <div className={s.resourceCard}>
            <div className={s.resourceTitle}>📖 Quotes Library</div>
            <input
              type="text"
              className={s.searchInput}
              placeholder="Search: anxiety, hope, calm..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <div className={s.quotesList}>
              {filteredQuotes.slice(0, 5).map((q, i) => (
                <div key={i} className={s.quoteItem}>
                  <div className={s.quoteItemText}>«{q.text}»</div>
                  {q.author && <div className={s.quoteItemAuthor}>— {q.author}</div>}
                  <div className={s.quoteTags}>{q.tags.join(' · ')}</div>
                </div>
              ))}
            </div>
          </div>

          <div className={s.emergencyNote}>
            <div className={s.emergencyIcon}>⚠️</div>
            <div className={s.emergencyText}>
              If you're in a crisis — please reach out to a professional.<br />
              📞 Russia helpline: <strong>8-800-2000-122</strong><br />
              📞 Kazakhstan: <strong>150</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}