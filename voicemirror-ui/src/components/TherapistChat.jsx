import { useState, useRef, useEffect } from 'react'
import s from '../styles/TherapistChat.module.css'

export default function TherapistChat({ emotion, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [quote, setQuote] = useState(null)
  const [grounding, setGrounding] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)

  const quotesLibrary = [
    { text: "Всё, что происходит, происходит в нужное время.", author: "Лао-цзы", tags: ["спокойствие", "принятие"] },
    { text: "Ты не обязан быть совершенным. Ты обязан быть собой.", author: "Будда", tags: ["самопринятие", "спокойствие"] },
    { text: "И это пройдёт.", author: "Соломон", tags: ["надежда", "утешение"] },
    { text: "Тревога — это волна. Она приходит и уходит. Ты — океан.", author: "Мать Тереза", tags: ["тревога", "спокойствие"] },
    { text: "Счастье можно найти даже в самые тёмные времена, если не забывать зажигать свет.", author: "Альбус Дамблдор", tags: ["надежда", "счастье"] },
    { text: "Ты гораздо сильнее, чем думаешь.", author: "", tags: ["уверенность", "поддержка"] },
  ]

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/chat')
    ws.onopen = () => console.log('Connected to therapist')
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'response') {
        setMessages(prev => [...prev, { role: 'assistant', content: data.text }])
        setIsTyping(false)
        if (data.quote) setQuote(data.quote)
        if (data.grounding) setGrounding(data.grounding)
      }
    }
    wsRef.current = ws
    
    setTimeout(() => {
      setMessages([{ 
        role: 'assistant', 
        content: `Привет! Я здесь, чтобы поддержать тебя. Расскажи, что у тебя на душе? 💙`
      }])
    }, 500)
    
    return () => ws.close()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current) return
    
    setMessages(prev => [...prev, { role: 'user', content: input }])
    wsRef.current.send(JSON.stringify({ text: input, emotion: emotion }))
    setInput('')
    setIsTyping(true)
  }

  const handleKeyPress = (e) => {
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
        <button className={s.closeBtn} onClick={onClose}>✕</button>
      </div>
      
      <div className={s.mainGrid}>
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
                <div className={s.typing}>
                  <span>🧠</span>
                  <span className={s.typingDots}>...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className={s.inputArea}>
            <textarea
              className={s.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Расскажи, что чувствуешь..."
              rows={2}
            />
            <button className={s.sendBtn} onClick={sendMessage} disabled={!input.trim()}>
              Отправить →
            </button>
          </div>
        </div>
        
        <div className={s.resourcesArea}>
          {quote && (
            <div className={s.resourceCard}>
              <div className={s.resourceTitle}>✨ Цитата дня</div>
              <div className={s.quoteText}>{quote}</div>
            </div>
          )}
          
          {grounding && (
            <div className={s.resourceCard}>
              <div className={s.resourceTitle}>🌿 Техника заземления</div>
              <div className={s.groundingText}>{grounding}</div>
            </div>
          )}
          
          <div className={s.resourceCard}>
            <div className={s.resourceTitle}>📖 Библиотека цитат</div>
            <input 
              type="text" 
              className={s.searchInput}
              placeholder="Поиск по теме (тревога, спокойствие, надежда...)"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
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
              Если ты в кризисной ситуации — пожалуйста, обратись к профессионалу.<br />
              📞 Горячая линия психологической помощи: <strong>8-800-333-44-34</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}