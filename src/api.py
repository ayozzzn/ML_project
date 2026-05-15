import os
import tempfile
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio

from src.inference import MindVoicePredictor

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

predictor: Optional[MindVoicePredictor] = None
voice_therapist = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, voice_therapist
    print("⏳ Loading MindVoice models...")
    try:
        loop = asyncio.get_event_loop()
        predictor = await loop.run_in_executor(None, MindVoicePredictor)
        print("✅ Models loaded")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")

    # VoiceTherapist загружается отдельно (Whisper тяжёлый)
    try:
        from voice_therapist import VoiceTherapist
        voice_therapist = await asyncio.get_event_loop().run_in_executor(None, VoiceTherapist)
        print("✅ VoiceTherapist (Whisper) loaded")
    except Exception as e:
        print(f"ℹ️  VoiceTherapist not available: {e}")

    yield


app = FastAPI(
    title="MindVoice API",
    description="Speech Emotion Recognition backend for MindVoice",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    emotion: str
    probabilities: Dict[str, float]
    confidence: float
    low_confidence: bool = False
    explanation: List[str]
    model_used: Optional[str] = None
    inference_time_sec: float = 0.0
    advice: Optional[str] = None
    quote: Optional[str] = None
    grounding: Optional[str] = None


class ChatRequest(BaseModel):
    emotion: str
    message: str
    history: List[Dict[str, str]] = []


class VoiceChatResponse(BaseModel):
    # Что пользователь сказал (транскрипт)
    user_text: str
    # Эмоция из голоса
    emotion: str
    confidence: float
    emotion_probabilities: Dict[str, float]
    # Ответ психолога
    therapist_response: str
    quote: Optional[str] = None
    grounding: Optional[str] = None
    # Метаданные сессии
    conversation_length: int = 0
    whisper_available: bool = False


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "MindVoice API v2.2 — твой эмоциональный компаньон"}


@app.get("/health")
def health():
    loaded = predictor is not None
    return {
        "status": "ok" if loaded else "models_not_loaded",
        "models_loaded": loaded,
        "wav2vec_available": loaded and getattr(predictor, "use_wav2vec", False),
        "ai_psychologist_available": loaded and predictor.ai_psychologist is not None,
        "whisper_available": voice_therapist is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), note: str = ""):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: predictor.predict_with_advice(tmp_path, user_note=note)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return PredictResponse(
        emotion=result["emotion"],
        probabilities=result["probabilities"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        advice=result.get("advice"),
        quote=result.get("quote"),
        grounding=result.get("grounding"),
        low_confidence=result.get("low_confidence", False),
        inference_time_sec=result.get("inference_time_sec", 0.0),
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Текстовый чат с психологом (без голоса)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: predictor.chat_with_therapist(
            emotion=request.emotion,
            user_message=request.message,
            history=request.history,
        )
    )
    return result


@app.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    file: UploadFile = File(...),
    language: str = "ru",
    session_id: str = "default",
):
    """
    Главный эндпоинт голосового чата с психологом.

    Pipeline:
      1. Сохраняем аудио во временный файл
      2. ПАРАЛЛЕЛЬНО: Whisper транскрибирует речь + emotion model определяет эмоцию
      3. LLM получает И текст И эмоцию → генерирует живой ответ психолога
      4. Возвращаем транскрипт, эмоцию, ответ
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    suffix = Path(file.filename or "audio.webm").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()

        if voice_therapist is not None:
            # ── Полный пайплайн: Whisper + эмоция + LLM ──────────────────────
            result = await loop.run_in_executor(
                None,
                lambda: voice_therapist.process_audio(tmp_path, language=language)
            )

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            return VoiceChatResponse(
                user_text=result["user_text"],
                emotion=result["emotion"],
                confidence=result["confidence"],
                emotion_probabilities=result["emotion_probabilities"],
                therapist_response=result["therapist_response"],
                quote=result.get("quote"),
                grounding=result.get("grounding"),
                conversation_length=result.get("conversation_length", 0),
                whisper_available=True,
            )

        else:
            # ── Fallback: только эмоция (без транскрипта) ─────────────────────
            emotion_result = await loop.run_in_executor(
                None,
                lambda: predictor.predict(tmp_path)
            )

            if "error" in emotion_result:
                raise HTTPException(status_code=400, detail=emotion_result["error"])

            emotion = emotion_result["emotion"]

            # Чат с психологом только на основе эмоции
            chat_result = await loop.run_in_executor(
                None,
                lambda: predictor.chat_with_therapist(
                    emotion=emotion,
                    user_message=f"Пользователь отправил голосовое. Эмоция: {emotion}, уверенность: {emotion_result['confidence']}%. Дай поддерживающий ответ.",
                    history=[],
                )
            )

            return VoiceChatResponse(
                user_text="",  # Whisper недоступен
                emotion=emotion,
                confidence=emotion_result["confidence"],
                emotion_probabilities=emotion_result["probabilities"],
                therapist_response=chat_result.get("response", "Я слышу тебя."),
                quote=chat_result.get("quote"),
                grounding=chat_result.get("grounding"),
                conversation_length=0,
                whisper_available=False,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/voice-chat/reset")
async def reset_voice_session(session_id: str = "default"):
    """Сбросить историю разговора (начать новую сессию)."""
    if voice_therapist is not None:
        voice_therapist.reset_conversation()
    return {"status": "ok", "message": "Conversation reset"}


@app.get("/voice-chat/summary")
async def voice_session_summary(session_id: str = "default"):
    """Получить сводку текущей сессии — доминирующая эмоция, тренд."""
    if voice_therapist is None:
        return {"has_history": False, "whisper_available": False}
    summary = voice_therapist.get_conversation_summary()
    summary["whisper_available"] = True
    return summary


@app.get("/emotions")
def get_emotions():
    emotions_ru = {
        "neutral": "Нейтральный",
        "calm": "Спокойный",
        "happy": "Радостный",
        "sad": "Грустный",
        "angry": "Злой",
        "anxiety": "Тревожный",
    }
    if predictor is None:
        return {"emotions": list(emotions_ru.keys()), "emotions_ru": emotions_ru}
    classes = list(predictor.label_encoder.classes_)
    return {
        "emotions": classes,
        "emotions_ru": {k: emotions_ru.get(k, k) for k in classes},
    }