import numpy as np
import librosa
import noisereduce as nr
import joblib
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional

MODELS_DIR = Path(__file__).parent.parent / "models"
SAMPLE_RATE = 16000
N_MFCC = 40


def preprocess_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = nr.reduce_noise(y=y, sr=sr, stationary=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        features[f"delta_mfcc_{i}_mean"] = float(np.mean(delta_mfcc[i]))
        features[f"delta2_mfcc_{i}_mean"] = float(np.mean(delta2_mfcc[i]))

    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[f0 > 0]
    features["pitch_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    features["pitch_std"] = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    features["pitch_range"] = float(np.ptp(f0_voiced)) if len(f0_voiced) > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_{i}_mean"] = float(np.mean(contrast[i]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"] = float(np.std(chroma))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features["mel_mean"] = float(np.mean(mel_db))
    features["mel_std"] = float(np.std(mel_db))

    non_silent = librosa.effects.split(y, top_db=20)
    speaking_frames = sum(end - start for start, end in non_silent)
    features["speaking_rate"] = speaking_frames / len(y) if len(y) > 0 else 0.0
    features["pause_ratio"] = 1.0 - features["speaking_rate"]
    features["num_pauses"] = float(len(non_silent))

    return features


class MindVoicePredictor:

    def __init__(self):
        self.use_wav2vec = False
        self.ai_psychologist = None
        self._load_models()
        self._load_ai_psychologist()

    def _load_models(self):
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        self.gbm = joblib.load(MODELS_DIR / "gbm_for_shap.pkl")
        self.shap_explainer = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        self.feature_names_human = joblib.load(MODELS_DIR / "feature_names_human.pkl")

        import pandas as pd
        sample = pd.read_csv(
            Path(__file__).parent.parent / "processed" / "features_ml.csv",
            nrows=1
        )
        self.feature_cols = [
            c for c in sample.columns
            if c not in ["emotion", "language", "dataset"]
        ]
        print(f"✅ Inference: loaded {len(self.feature_cols)} feature columns")

        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
            wav2vec_path = MODELS_DIR / "wav2vec2_emotion"
            if wav2vec_path.exists():
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
                self.wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    str(wav2vec_path)
                )
                self.wav2vec_model.eval()
                self.use_wav2vec = True
                print("✅ Wav2Vec2 loaded")
            else:
                print("ℹ️  Wav2Vec2 model folder not found — using GBM fallback")
        except Exception as e:
            print(f"⚠️  Wav2Vec2 not available: {e}")

    def _load_ai_psychologist(self):
        try:
            from src.ai_therapist import AIPsychologist
            self.ai_psychologist = AIPsychologist()
            print("✅ AI Psychologist ready")
        except Exception as e:
            print(f"⚠️ AI Psychologist not available: {e}")

    def predict(self, audio_path: str) -> Dict:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        y = preprocess_audio(y, sr)

        if len(y) < SAMPLE_RATE * 0.5:
            return {"error": "Audio too short. Please record at least 1 second."}

        raw_features = extract_features(y, sr)

        feat_vector = np.array(
            [raw_features.get(c, 0.0) for c in self.feature_cols],
            dtype=np.float32
        )
        feat_scaled = self.scaler.transform(feat_vector.reshape(1, -1))

        if self.use_wav2vec:
            emotion, probs = self._predict_wav2vec(y)
        else:
            probs_raw = self.gbm.predict_proba(feat_scaled)[0]
            pred_idx = int(np.argmax(probs_raw))
            emotion = self.label_encoder.classes_[pred_idx]
            probs = {
                cls: float(p)
                for cls, p in zip(self.label_encoder.classes_, probs_raw)
            }

        confidence = round(probs.get(emotion, 0.0) * 100, 1)
        explanation = self._explain(feat_scaled[0], emotion)

        return {
            "emotion": emotion,
            "probabilities": probs,
            "confidence": confidence,
            "explanation": explanation,
        }

    def predict_with_advice(self, audio_path: str, user_note: str = "", conversation_history: List[Dict] = None) -> Dict:
        result = self.predict(audio_path)
        
        if "error" in result:
            return result
        
        advice_result = self.chat_with_therapist(
            emotion=result["emotion"],
            user_message=user_note if user_note else "I just recorded my voice. Can you support me?",
            history=conversation_history
        )
        
        result["advice"] = advice_result.get("response", "")
        result["quote"] = advice_result.get("quote")
        result["grounding"] = advice_result.get("grounding")
        
        return result

    def chat_with_therapist(self, emotion: str, user_message: str, history: List[Dict] = None) -> Dict:
        if self.ai_psychologist:
            return self.ai_psychologist.generate_response(emotion, user_message, history)

        fallback_responses = {
            "sad": "Мне жаль, что тебе сейчас грустно. Расскажи подробнее — я здесь, чтобы выслушать и поддержать. 💙",
            "anxiety": "Тревога может быть очень тяжёлой. Но сейчас, в этом моменте, ты в безопасности. Давай попробуем подышать вместе? 🌊",
            "angry": "Я слышу твоё раздражение. Ты имеешь на него полное право. Хочешь рассказать, что тебя задело? 🔥",
            "happy": "Я так рад, что ты делишься этой радостью! Расскажи ещё — что именно сделало твой день лучше? ✨",
            "calm": "Это прекрасное состояние. Посиди в нём ещё немного. Ты заслужил этот покой. 🧘",
            "neutral": "Спасибо, что делишься. Как ты себя чувствуешь прямо сейчас, в теле, в мыслях? 🌿"
        }
        
        return {
            "response": fallback_responses.get(emotion.lower(), fallback_responses["neutral"]),
            "quote": "✨ *Будь добр к себе сегодня*",
            "grounding": None
        }

    def _predict_wav2vec(self, y: np.ndarray) -> Tuple[str, Dict]:
        inputs = self.wav2vec_processor(
            y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.wav2vec_model(**inputs).logits
        probs_t = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs_t.argmax().item())
        emotion = self.label_encoder.classes_[pred_idx]
        probs = {
            cls: float(p)
            for cls, p in zip(self.label_encoder.classes_, probs_t.numpy())
        }
        return emotion, probs

    def _explain(self, feat_scaled: np.ndarray, predicted_emotion: str, top_n: int = 3) -> List[str]:
        sv = self.shap_explainer.shap_values(feat_scaled.reshape(1, -1))

        try:
            class_idx = list(self.label_encoder.classes_).index(predicted_emotion)
        except ValueError:
            class_idx = 0

        if isinstance(sv, list):
            shap_vals = sv[class_idx][0]
        elif sv.ndim == 3:
            shap_vals = sv[0, :, class_idx]
        else:
            shap_vals = sv[0]

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

        lines = []
        for i in top_idx:
            feat = self.feature_cols[i]
            human = self.feature_names_human.get(feat, feat.replace("_", " "))
            val = shap_vals[i]
            direction = "higher than usual" if val > 0 else "lower than usual"
            strength = "slightly" if abs(val) < 0.05 else "noticeably"
            lines.append(f"Your {human} was {strength} {direction}")

        return lines