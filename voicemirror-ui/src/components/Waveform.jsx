import { useEffect, useRef } from 'react'
import s from '../styles/Waveform.module.css'

const BAR_COUNT = 48

export default function Waveform({ active }) {
  const barsRef = useRef([])
  const frameRef = useRef(null)
  const tRef = useRef(0)

  useEffect(() => {
    if (!active) {
      cancelAnimationFrame(frameRef.current)
      barsRef.current.forEach((b, i) => {
        if (b) {
          b.style.height = '4px'
          b.style.opacity = '0.25'
        }
      })
      return
    }

    const animate = () => {
      tRef.current += 0.055
      const t = tRef.current
      barsRef.current.forEach((b, i) => {
        if (!b) return
        const wave =
          Math.abs(Math.sin(t + i * 0.32)) * 30 +
          Math.abs(Math.sin(t * 1.7 + i * 0.18)) * 20 +
          Math.random() * 8 + 4
        b.style.height = `${wave}px`
        b.style.opacity = '1'
      })
      frameRef.current = requestAnimationFrame(animate)
    }
    frameRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frameRef.current)
  }, [active])

  return (
    <div className={s.container}>
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <div
          key={i}
          ref={el => barsRef.current[i] = el}
          className={s.bar}
          style={{
            animationDelay: active ? '0ms' : `${i * 42}ms`,
            height: '4px',
            opacity: 0.25,
          }}
        />
      ))}
    </div>
  )
}