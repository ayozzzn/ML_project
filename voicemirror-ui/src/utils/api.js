const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`)
    if (!res.ok) return { ok: false, models_loaded: false }
    return await res.json()
  } catch {
    return { ok: false, models_loaded: false }
  }
}

export async function uploadAudio(file, note = '') {
  const form = new FormData()
  form.append('file', file)
  if (note) form.append('note', note)

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: form,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }

    const data = await res.json()

    return {
      emotion:            data.emotion,
      probabilities:      data.probabilities,
      confidence:         data.confidence,
      explanation:        data.explanation,
      advice:             data.advice,
      quote:              data.quote,
      grounding:          data.grounding,
      low_confidence:     data.low_confidence ?? false,
      lang_hint:          data.lang_hint ?? 'en',
      inference_time_sec: data.inference_time_sec ?? 0,
      model_used:         data.model_used,
      demo:               false,
    }
  } catch (err) {
    console.warn('⚠️ FastAPI unreachable — demo mode:', err.message)
    return _demoResult()
  }
}

function _demoResult() {
  const emotions = ['happy', 'calm', 'neutral', 'sad', 'angry', 'anxiety']
  const raw = emotions.map(() => Math.random() * Math.random())
  const sum = raw.reduce((a, b) => a + b, 0)
  const probs = Object.fromEntries(emotions.map((e, i) => [e, raw[i] / sum]))
  const emotion = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0]
  const confidence = Math.round(probs[emotion] * 100)

  const demoAdvice = {
    sad:     "Мне жаль, что тебе сейчас грустно. Помни, что это чувство временное. Ты сильный человек, и эта боль пройдёт. Хочешь поговорить о том, что тебя беспокоит? 💙",
    anxiety: "Твоя тревога — это сигнал, что ты заботишься о себе. Но сейчас ты в безопасности. Давай сделаем глубокий вдох вместе... 🌊",
    angry:   "Гнев — это энергия. Ты можешь направить её во что-то созидательное или просто выдохнуть. Я здесь, чтобы выслушать. 🔥",
    happy:   "Твоя радость заразительна! Поделись ей с кем-нибудь сегодня, и она станет ещё больше. ✨",
    calm:    "Это прекрасное состояние. Посиди в тишине ещё немного. Ты заслужил этот покой. 🧘",
    neutral: "Спасибо, что делишься. Как ты себя чувствуешь прямо сейчас? 🌿",
  }

  return {
    emotion,
    probabilities: probs,
    confidence,
    explanation: [
      'Ваш средний тон голоса был заметно ниже обычного',
      'Длительность пауз была немного выше обычного',
      'Энергия голоса была заметно ниже обычного',
    ],
    advice:             demoAdvice[emotion] || demoAdvice.neutral,
    quote:              "✨ *Будь добр к себе сегодня*",
    grounding:          "🌿 Сделай 3 глубоких вдоха. Вдох — спокойствие, выдох — напряжение.",
    low_confidence:     false,
    lang_hint:          'en',
    inference_time_sec: 0,
    model_used:         'demo',
    demo:               true,
  }
}