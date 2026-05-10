import s from '../styles/AboutPage.module.css'

const CARDS = [
  {
    icon: '🧠',
    title: 'PRIMARY MODEL',
    color: '#5b7fff',
    body: 'Fine-tuned Wav2Vec2 by Facebook AI, pre-trained on 960h of speech. Supports both English and Russian (RESD dataset).'
  },
  {
    icon: '📡',
    title: 'BASELINE MODEL',
    color: '#4dffa0',
    body: '1D CNN + Bidirectional LSTM trained on 40-coefficient MFCC features. Lightweight, fast, and interpretable.'
  },
  {
    icon: '🔬',
    title: 'EXPLAINABILITY',
    color: '#c96cff',
    body: 'SHAP (SHapley Additive Explanations) shows exactly which acoustic features drove each prediction — fully transparent.'
  },
  {
    icon: '🌍',
    title: 'MULTILINGUAL',
    color: '#ffd84a',
    body: 'Trained on English (RAVDESS, CREMA-D, TESS) and Russian (RESD) datasets. Cross-lingual generalization tested.'
  },
]

const DATASETS = [
  { name: 'RESD',    lang: 'Russian', samples: '~4,000', color: '#ff6060' },
  { name: 'RAVDESS', lang: 'English', samples: '1,440',  color: '#5b7fff' },
  { name: 'CREMA-D', lang: 'English', samples: '7,442',  color: '#4dffa0' },
  { name: 'TESS',    lang: 'English', samples: '2,800',  color: '#ffd84a' },
]

const PIPELINE = [
  { step: '01', label: 'AUDIO INPUT',      sub: 'WAV / MP3 upload or mic recording', color: '#5b7fff' },
  { step: '02', label: 'PREPROCESSING',    sub: 'Noise reduction · trim · 16kHz',     color: '#7fa3ff' },
  { step: '03', label: 'FEATURE EXTRACT',  sub: 'MFCC · Pitch · Energy · ZCR',        color: '#4dffa0' },
  { step: '04', label: 'MODEL INFERENCE',  sub: 'Wav2Vec2 or CNN+LSTM',                color: '#c96cff' },
  { step: '05', label: 'SHAP EXPLAIN',     sub: 'Acoustic feature attribution',        color: '#ffd84a' },
  { step: '06', label: 'EMOTION OUTPUT',   sub: '6 classes · confidence score',        color: '#ff6060' },
]

export default function AboutPage() {
  return (
    <div className={s.root}>
      <div className={s.hero}>
        <div className={s.heroLabel}>THE TECHNOLOGY</div>
        <h1 className={s.heroTitle}>
          Built for<br />
          <span className={s.heroAccent}>understanding you</span>
        </h1>
        <p className={s.heroSub}>
          VoiceMirror uses state-of-the-art speech analysis to detect your
          emotional state from your voice — not your words.
        </p>
      </div>

      <div className={s.pipelineCard}>
        <div className={s.sectionLabel}>END-TO-END PIPELINE</div>
        <div className={s.pipeline}>
          {PIPELINE.map((p, i) => (
            <div key={i} className={s.pipelineStep}>
              <div className={s.pipelineNum} style={{ color: p.color, borderColor: p.color + '40' }}>
                {p.step}
              </div>
              <div className={s.pipelineLabel}>{p.label}</div>
              <div className={s.pipelineSub}>{p.sub}</div>
              {i < PIPELINE.length - 1 && (
                <div className={s.pipelineArrow} style={{ color: p.color }}>→</div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className={s.cardsGrid}>
        {CARDS.map((c, i) => (
          <div key={i} className={s.card} style={{ '--card-color': c.color }}>
            <div className={s.cardIcon}>{c.icon}</div>
            <div className={s.cardTitle}>{c.title}</div>
            <div className={s.cardBody}>{c.body}</div>
            <div className={s.cardAccent} />
          </div>
        ))}
      </div>

      <div className={s.datasetsCard}>
        <div className={s.sectionLabel}>TRAINING DATASETS</div>
        <div className={s.datasetTable}>
          <div className={s.tableHeader}>
            <span>DATASET</span><span>LANGUAGE</span><span>SAMPLES</span><span>STATUS</span>
          </div>
          {DATASETS.map((d, i) => (
            <div key={i} className={s.tableRow}>
              <span className={s.dsName} style={{ color: d.color }}>{d.name}</span>
              <span className={s.dsLang}>{d.lang}</span>
              <span className={s.dsSamples}>{d.samples}</span>
              <span className={s.dsStatus}>✓ LOADED</span>
            </div>
          ))}
        </div>
      </div>

      <div className={s.emotionsCard}>
        <div className={s.sectionLabel}>DETECTED EMOTIONS</div>
        <div className={s.emotionGrid}>
          {[
            { emoji:'😊', name:'HAPPY',   color:'#ffd84a', desc:'Joy, positivity, excitement' },
            { emoji:'😌', name:'CALM',    color:'#4dffa0', desc:'Relaxed, at ease, peaceful' },
            { emoji:'😐', name:'NEUTRAL', color:'#80b8ff', desc:'Balanced, no strong emotion' },
            { emoji:'😢', name:'SAD',     color:'#5b7fff', desc:'Low mood, grief, tiredness' },
            { emoji:'😠', name:'ANGRY',   color:'#ff6060', desc:'Frustration, irritation' },
            { emoji:'😰', name:'ANXIETY', color:'#c96cff', desc:'Stress, worry, fear' },
          ].map((e, i) => (
            <div key={i} className={s.emotionItem} style={{ '--em': e.color }}>
              <div className={s.emotionEmoji}>{e.emoji}</div>
              <div className={s.emotionName}>{e.name}</div>
              <div className={s.emotionDesc}>{e.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div className={s.disclaimer}>
        <span className={s.disclaimerIcon}>⚠</span>
        MindVoice is not a medical device and does not diagnose any condition.
        It is a personal wellness tool for self-reflection only. If you are
        experiencing difficulties, please consult a mental health professional.
      </div>
    </div>
  )
}