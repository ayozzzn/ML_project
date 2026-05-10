import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

DB_PATH = Path(__file__).parent.parent / "mindvoice.db"

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            emotion     TEXT NOT NULL,
            prob_neutral  REAL DEFAULT 0,
            prob_calm     REAL DEFAULT 0,
            prob_happy    REAL DEFAULT 0,
            prob_sad      REAL DEFAULT 0,
            prob_angry    REAL DEFAULT 0,
            prob_anxiety  REAL DEFAULT 0,
            explanation   TEXT,
            note          TEXT,
            duration_sec  REAL
        )
    """)
    conn.commit()
    conn.close()

def save_entry(emotion: str, probabilities: Dict[str, float],
               explanation: str = "", note: str = "", duration_sec: float = 0.0):
    conn = get_connection()
    conn.execute("""
        INSERT INTO entries
            (timestamp, emotion, prob_neutral, prob_calm, prob_happy,
             prob_sad, prob_angry, prob_anxiety, explanation, note, duration_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        emotion,
        probabilities.get("neutral",  0.0),
        probabilities.get("calm",     0.0),
        probabilities.get("happy",    0.0),
        probabilities.get("sad",      0.0),
        probabilities.get("angry",    0.0),
        probabilities.get("anxiety",  0.0),
        explanation,
        note,
        duration_sec
    ))
    conn.commit()
    conn.close()

def get_recent_entries(days: int = 30) -> pd.DataFrame:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM entries WHERE timestamp >= ? ORDER BY timestamp DESC",
        conn,
        params=(cutoff,)
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def compute_trends(df: pd.DataFrame) -> Dict:
    if df.empty or len(df) < 2:
        return {}

    df = df.sort_values("timestamp")
    now = df["timestamp"].max()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)

    this_week = df[df["timestamp"] >= week_ago]
    last_week = df[(df["timestamp"] >= two_weeks_ago) & (df["timestamp"] < week_ago)]

    if this_week.empty or last_week.empty:
        return {}

    emotions = ["neutral", "calm", "happy", "sad", "angry", "anxiety"]
    trends = {}
    for em in emotions:
        col = f"prob_{em}"
        if col in df.columns:
            curr = this_week[col].mean()
            prev = last_week[col].mean()
            if prev > 0:
                change_pct = ((curr - prev) / prev) * 100
            else:
                change_pct = 0.0
            trends[em] = {
                "current_avg": round(curr, 3),
                "previous_avg": round(prev, 3),
                "change_pct": round(change_pct, 1)
            }
    return trends

def generate_trend_message(trends: Dict) -> str:
    if not trends:
        return "Not enough data yet for trend analysis. Keep recording!"

    messages = []
    anxiety_trend = trends.get("anxiety", {})
    if anxiety_trend:
        pct = anxiety_trend["change_pct"]
        if pct > 20:
            messages.append(f"⚠️ Your anxiety indicators increased by {pct:.0f}% this week. Consider taking breaks or talking to someone close.")
        elif pct < -20:
            messages.append(f"✅ Your anxiety indicators decreased by {abs(pct):.0f}% this week. You seem to be doing better!")

    happy_trend = trends.get("happy", {})
    if happy_trend:
        pct = happy_trend["change_pct"]
        if pct > 15:
            messages.append(f"😊 Your positivity increased by {pct:.0f}% this week — keep it up!")
        elif pct < -15:
            messages.append(f"😕 Your positivity decreased by {abs(pct):.0f}% this week.")

    sad_trend = trends.get("sad", {})
    if sad_trend and sad_trend["change_pct"] > 25:
        messages.append(f"💙 Sadness indicators are up {sad_trend['change_pct']:.0f}%. Be kind to yourself.")

    if not messages:
        messages.append("📊 Your emotional state has been relatively stable this week.")

    return "\n".join(messages)

init_db()