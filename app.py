import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import tempfile
import os
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from database import save_entry, get_recent_entries, compute_trends, generate_trend_message

st.set_page_config(
    page_title="MindVoice",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4A90D9;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #4A90D9;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .explanation-box {
        background: #EEF5FB;
        border: 1px solid #B8D4F0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def show_results(result, note=""):
    """Функция для отображения результатов анализа"""
    emotion = result["emotion"]
    probs = result["probabilities"]
    expl = result.get("explanation", "")

    EMOTION_EMOJI = {
        "happy": "😊", "calm": "😌", "neutral": "😐",
        "sad": "😢", "angry": "😠", "anxiety": "😰"
    }
    emoji = EMOTION_EMOJI.get(emotion, "🎯")

    st.success("✅ Analysis complete!")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"""
        <div class="emotion-card">
            <div style="font-size: 3rem">{emoji}</div>
            <div style="font-size: 1.8rem; font-weight: bold; text-transform: capitalize">{emotion}</div>
            <div style="font-size: 0.9rem; opacity: 0.8">Primary emotion detected</div>
        </div>
        """, unsafe_allow_html=True)

        confidence = probs.get(emotion, 0) * 100
        st.metric("Confidence", f"{confidence:.1f}%")

        save_entry(
            emotion=emotion,
            probabilities=probs,
            explanation=expl if isinstance(expl, str) else str(expl),
            note=note,
        )
        st.caption("📁 Entry saved to your diary")

    with col2:
        df_probs = pd.DataFrame([
            {"Emotion": k.capitalize(), "Probability": v * 100}
            for k, v in sorted(probs.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(
            df_probs, x="Probability", y="Emotion",
            orientation="h",
            color="Probability",
            color_continuous_scale="Blues",
            title="Emotion Probabilities (%)"
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if expl:
        st.markdown("### 🔍 Why this prediction?")
        if isinstance(expl, list):
            expl_text = "<br>".join(expl)
        else:
            expl_text = expl.replace(chr(10), "<br>")
        st.markdown(f'<div class="explanation-box">{expl_text}</div>', unsafe_allow_html=True)
        st.caption("Explanation powered by SHAP (SHapley Additive Explanations)")

    if note:
        st.markdown(f"**Your note:** *{note}*")

@st.cache_resource
def load_predictor():
    try:
        from inference import MindVoicePredictor
        return MindVoicePredictor()
    except Exception as e:
        return None

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=60)
    st.title("MindVoice")
    st.markdown("*Your emotional self-reflection assistant*")
    st.divider()
    page = st.radio("Navigate", ["🎙️ Record & Analyze", "📊 My Trends", "ℹ️ About"])
    st.divider()
    st.caption("Privacy: All data is stored locally on your device.")

if page == "🎙️ Record & Analyze":
    st.markdown('<p class="main-header">🎙️ MindVoice</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Record a short voice diary about your day. We\'ll analyze your emotional state.</p>', unsafe_allow_html=True)

    predictor = load_predictor()
    if predictor is None:
        st.warning("⚠️ Models not loaded yet. Using demo mode with random predictions.")
        st.info("For real predictions, make sure you trained all models first.")

    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record (if mic enabled)"])

    with tab1:
        st.markdown("**Upload a voice recording** (.wav, .mp3, .ogg, .flac)")
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "ogg", "flac", "m4a", "webm"])
        note = st.text_area("📝 Optional: Add a note about your day", placeholder="Today I felt...", height=80)

        if uploaded_file and st.button("🔍 Analyze Emotion", type="primary"):
            with st.spinner("Analyzing your voice..."):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if predictor:
                    result = predictor.predict(tmp_path)
                else:
                    emotions = ["neutral", "calm", "happy", "sad", "angry", "anxiety"]
                    probs = np.random.dirichlet(np.ones(6))
                    result = {
                        "emotion": emotions[np.argmax(probs)],
                        "probabilities": dict(zip(emotions, probs.tolist())),
                        "explanation": [
                            "Your **average pitch** was noticeably lower than usual",
                            "Your **pause duration** was slightly higher than usual",
                            "Your **voice energy** was noticeably lower than usual"
                        ],
                    }

                os.unlink(tmp_path)

            if "error" in result:
                st.error(result["error"])
            else:
                show_results(result, note)

    with tab2:
        st.info("🎤 Browser microphone recording requires HTTPS or localhost. Use the Upload tab for now.")
        st.markdown("**Tip:** Use your phone's Voice Memo app, save as .m4a or .wav, then upload above.")

elif page == "📊 My Trends":
    st.markdown('<p class="main-header">📊 Emotional Trends</p>', unsafe_allow_html=True)

    days = st.slider("Show last N days", 7, 90, 30)
    df = get_recent_entries(days=days)

    if df.empty:
        st.info("No entries yet. Record your first voice diary on the previous page!")
    else:
        st.markdown(f"**{len(df)} entries** in the selected period")

        trends = compute_trends(df)
        trend_msg = generate_trend_message(trends)
        st.markdown("### 💡 Weekly Summary")
        for line in trend_msg.split("\n"):
            st.markdown(line)

        st.divider()

        st.markdown("### Emotion Over Time")
        emotion_cols = ["prob_neutral", "prob_calm", "prob_happy", "prob_sad", "prob_angry", "prob_anxiety"]
        existing_cols = [c for c in emotion_cols if c in df.columns]
        if existing_cols:
            df_plot = df[["timestamp"] + existing_cols].copy()
            df_plot = df_plot.sort_values("timestamp")

            rename = {c: c.replace("prob_", "").capitalize() for c in existing_cols}
            df_plot = df_plot.rename(columns=rename)

            fig = go.Figure()
            colors = {
                "Neutral": "#95a5a6", "Calm": "#27ae60", "Happy": "#f1c40f",
                "Sad": "#2980b9", "Angry": "#e74c3c", "Anxiety": "#8e44ad"
            }
            for em, color in colors.items():
                if em in df_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=df_plot["timestamp"], y=df_plot[em] * 100,
                        name=em, line=dict(color=color, width=2),
                        mode="lines+markers"
                    ))

            fig.update_layout(
                title="Emotion Probability Trend (%)",
                xaxis_title="Date",
                yaxis_title="Probability (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        if "prob_anxiety" in df.columns and df["prob_anxiety"].sum() > 0:
            st.markdown("### 🔴 Anxiety Index (7-day rolling average)")
            df_sorted = df.sort_values("timestamp").set_index("timestamp")
            rolling = df_sorted["prob_anxiety"].rolling("7D").mean() * 100
            fig2 = px.area(rolling.reset_index(), x="timestamp", y="prob_anxiety",
                           labels={"prob_anxiety": "Anxiety Index (%)", "timestamp": "Date"},
                           color_discrete_sequence=["#8e44ad"])
            fig2.update_layout(height=280)
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Distribution of Detected Emotions")
            counts = df["emotion"].value_counts().reset_index()
            counts.columns = ["Emotion", "Count"]
            fig3 = px.pie(counts, values="Count", names="Emotion",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown("### Raw Entries")
            display_df = df[["timestamp", "emotion", "note"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

else:
    st.markdown('<p class="main-header">ℹ️ About MindVoice</p>', unsafe_allow_html=True)
    st.markdown("""
    **MindVoice** is an emotional self-reflection tool that uses Speech Emotion Recognition (SER)
    to help you track your emotional state over time through short voice recordings.

    ### How it works
    1. **Record** a 1–3 minute voice diary entry about your day
    2. **AI analyzes** your voice — pitch, energy, pauses, speaking rate
    3. **View** your detected emotion + which acoustic features drove the prediction
    4. **Track** your emotional trends over days and weeks

    ### Models Used
    - **Primary**: Fine-tuned Wav2Vec2 (Facebook AI) — multilingual, English + Russian
    - **Baseline**: 1D CNN + LSTM on MFCC acoustic features
    - **Explainability**: SHAP (SHapley Additive Explanations)

    ### Datasets
    | Dataset | Language | Samples |
    |---------|----------|---------|
    | RESD    | Russian  | ~4,000  |
    | RAVDESS | English  | 1,440   |
    | CREMA-D | English  | 7,442   |
    | TESS    | English  | 2,800   |

    ### ⚠️ Disclaimer
    MindVoice is **not a medical device**. It does not diagnose any condition.
    It is a personal wellness tool for self-reflection only.
    If you're experiencing mental health difficulties, please consult a professional.

    ---
    Built with ❤️ using Python, PyTorch, HuggingFace Transformers, and Streamlit.
    """)