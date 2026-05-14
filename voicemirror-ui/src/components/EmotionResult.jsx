import { EMOTIONS } from '../utils/storage.js'
import s from '../styles/EmotionResult.module.css'

export default function EmotionResult({ result }) {
  const { emotion, probabilities: probs, confidence, explanation } = result
  const em = EMOTIONS[emotion] || EMOTIONS.neutral

  const sorted = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])

  return (
    <div className={s.root}>
      <div className={s.hero} style={{ '--em-color': em.color }}>
        <div className={s.heroGlow} />
        <div className={s.heroInner}>
          <div className={s.emoji}>{em.emoji}</div>
          <div className={s.emotionName}>{em.label}</div>
          <div className={s.emotionDesc}>{em.desc}</div>
          <div className={s.confidence}>
            <span className={s.confBar}>
              <span className={s.confFill} style={{ width: `${confidence}%`, background: em.color }} />
            </span>
            <span className={s.confNum}>{confidence}% confidence</span>
          </div>
        </div>
      </div>

      <div className={s.card}>
        <div className={s.cardLabel}>EMOTION DISTRIBUTION</div>
        <div className={s.probList}>
          {sorted.map(([key, val]) => {
            const e2 = EMOTIONS[key] || {}
            const pct = Math.round(val * 100)
            return (
              <div key={key} className={s.probRow}>
                <span className={s.probEmoji}>{e2.emoji}</span>
                <span className={s.probName}>{key.toUpperCase()}</span>
                <div className={s.probTrack}>
                  <div
                    className={s.probFill}
                    style={{ width: `${pct}%`, background: e2.color }}
                  />
                </div>
                <span className={s.probPct}>{pct}%</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className={s.card}>
        <div className={s.cardLabel}>WHY THIS PREDICTION <span className={s.shap}>SHAP</span></div>
        <div className={s.explList}>
          {explanation.map((line, i) => {
            const parts = line.split('**')
            return (
              <div key={i} className={s.explRow}>
                <span className={s.explDot} style={{ background: em.color }} />
                <span className={s.explText}>
                  {parts.map((p, j) =>
                    j % 2 === 1
                      ? <strong key={j} style={{ color: em.color }}>{p}</strong>
                      : p
                  )}
                </span>
              </div>
            )
          })}
        </div>
        <div className={s.shapNote}>
          Powered by SHapley Additive Explanations — shows which acoustic features drove the prediction
        </div>
      </div>
    </div>
  )
}