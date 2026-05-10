import os
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

from src.inference import MindVoicePredictor

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

app = FastAPI(
    title="MindVoice API",
    description="Speech Emotion Recognition backend for MindVoice with AI Psychologist",
    version="2.0.0"
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

predictor: Optional[MindVoicePredictor] = None


@app.on_event("startup")
async def startup_event():
    global predictor
    print("⏳ Loading MindVoice models...")
    try:
        predictor = MindVoicePredictor()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")


class PredictResponse(BaseModel):
    emotion: str
    probabilities: Dict[str, float]
    confidence: float
    explanation: List[str]
    advice: Optional[str] = None
    quote: Optional[str] = None
    grounding: Optional[str] = None


class ChatRequest(BaseModel):
    emotion: str
    message: str
    history: List[Dict[str, str]] = []


@app.get("/")
def root():
    return {"message": "MindVoice API v2.0 - Your emotional support companion"}


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
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run training first."
        )

    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predictor.predict_with_advice(tmp_path, user_note=note)
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
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    result = predictor.chat_with_therapist(
        emotion=request.emotion,
        user_message=request.message,
        history=request.history
    )
    
    return result


@app.get("/emotions")
def get_emotions():
    if predictor is None:
        return {"emotions": ["neutral", "calm", "happy", "sad", "angry", "anxiety"]}
    return {"emotions": list(predictor.label_encoder.classes_)}