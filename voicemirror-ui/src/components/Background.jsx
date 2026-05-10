import s from '../styles/Background.module.css'

export default function Background() {
  return (
    <div className={s.root} aria-hidden="true">
      <div className={`${s.orb} ${s.orb1}`} />
      <div className={`${s.orb} ${s.orb2}`} />
      <div className={`${s.orb} ${s.orb3}`} />
      <div className={`${s.orb} ${s.orb4}`} />
      <div className={s.grid} />
      <div className={s.scanline} />
    </div>
  )
}