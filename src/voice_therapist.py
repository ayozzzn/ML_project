import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter


class VoiceTherapist:

    def __init__(self):
        print("🎤 Initializing Voice Therapist...")

        self.asr_model = None
        self.whisper_available = False
        try:
            from faster_whisper import WhisperModel
            self.asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self.whisper_available = True
            print("  ✅ Whisper ASR loaded")
        except ImportError:
            print("  ⚠️  faster-whisper not installed")
        except Exception as e:
            print(f"  ⚠️  Whisper error: {e}")

        print("  - Loading Emotion Model...")
        from src.inference import MindVoicePredictor
        self.emotion_model = MindVoicePredictor()

        print("  - Loading AI Therapist...")
        from src.ai_therapist import AIPsychologist
        self.therapist = AIPsychologist()

        self.conversation_history: List[Dict[str, str]] = []
        self.emotion_history: List[str] = []

        print("✅ Voice Therapist ready!\n")

    def process_audio(self, audio_path: str, language: str = "ru", user_note: str = "") -> Dict:
        t0 = time.time()

        user_text = self._transcribe(audio_path, language)

        emotion_result = self.emotion_model.predict(audio_path)
        if "error" in emotion_result:
            return {**emotion_result, "whisper_available": self.whisper_available}

        detected_emotion = emotion_result["emotion"]
        self.emotion_history.append(detected_emotion)

        context = self._build_context(
            user_text=user_text,
            user_note=user_note,
            emotion=detected_emotion,
            emotion_result=emotion_result,
        )

        response = self.therapist.generate_response(
            emotion=detected_emotion,
            user_message=context,
            conversation_history=self.conversation_history,
            include_quote=True,
            include_grounding=True,
        )

        history_content = user_text if user_text else f"[voice, emotion: {detected_emotion}]"
        if user_note:
            history_content += f"\n[note: {user_note}]"
        self.conversation_history.append({"role": "user", "content": history_content})
        self.conversation_history.append({"role": "assistant", "content": response["response"]})

        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return {
            "user_text": user_text,
            "whisper_available": self.whisper_available,
            "emotion": detected_emotion,
            "confidence": emotion_result.get("confidence", 0),
            "emotion_probabilities": emotion_result.get("probabilities", {}),
            "explanation": emotion_result.get("explanation", []),
            "therapist_response": response["response"],
            "quote": response.get("quote"),
            "grounding": response.get("grounding"),
            "emotion_trend": self.emotion_history[-5:],
            "conversation_length": len(self.conversation_history) // 2,
            "inference_time_sec": round(time.time() - t0, 2),
        }

    def reset_session(self):
        self.conversation_history = []
        self.emotion_history = []
        print("🔄 Session reset")

    def get_session_summary(self) -> Dict:
        if not self.emotion_history:
            return {"has_history": False}
        emotion_counts = Counter(self.emotion_history)
        dominant = emotion_counts.most_common(1)[0][0]
        return {
            "has_history": True,
            "total_messages": len(self.conversation_history) // 2,
            "dominant_emotion": dominant,
            "emotion_distribution": dict(emotion_counts),
            "emotion_trend": self.emotion_history[-5:],
        }

    def _transcribe(self, audio_path: str, language: str) -> Optional[str]:
        if not self.asr_model:
            return None
        try:
            segments, _ = self.asr_model.transcribe(audio_path, language=language)
            text = " ".join(seg.text for seg in segments).strip()
            return text or None
        except Exception as e:
            print(f"⚠️ Whisper error: {e}")
            return None

    def _build_context(self, user_text: Optional[str], user_note: str, emotion: str, emotion_result: Dict) -> str:
        parts = []

        if user_text:
            parts.append(f"User said: «{user_text}»")
        else:
            parts.append(f"[Voice message, emotion: {emotion}, confidence: {round(emotion_result.get('confidence', 0))}%]")

        if user_note:
            parts.append(f"User note: «{user_note}»")

        if len(self.emotion_history) > 2:
            trend = " → ".join(self.emotion_history[-5:])
            parts.append(f"[Emotion trend: {trend}]")

        return "\n\n".join(parts)