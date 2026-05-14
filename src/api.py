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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("⏳ Loading MindVoice models...")
    try:
        loop = asyncio.get_event_loop()
        predictor = await loop.run_in_executor(None, MindVoicePredictor)
        print("✅ Models loaded")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
    yield
    # shutdown (если нужно что-то почистить)


app = FastAPI(
    title="MindVoice API",
    description="Speech Emotion Recognition backend for MindVoice",
    version="2.1.0",
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


@app.get("/")
def root():
    return {"message": "MindVoice API v2.1 — твой эмоциональный компаньон"}


@app.get("/health")
def health():
    loaded = predictor is not None
    return {
        "status": "ok" if loaded else "models_not_loaded",
        "models_loaded": loaded,
        "wav2vec_available": loaded and getattr(predictor, "use_wav2vec", False),
        "ai_psychologist_available": loaded and predictor.ai_psychologist is not None,
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
        lang_hint=result.get("lang_hint", "en"),
        inference_time_sec=result.get("inference_time_sec", 0.0),
    )


@app.post("/chat")
async def chat(request: ChatRequest):
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