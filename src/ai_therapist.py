import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import re

class AIPsychologist:
    """
    Нейросеть-психолог для поддержки пользователя.
    Использует Llama 3.2 или Qwen с инструкциями.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        model_name: 
            - "Qwen/Qwen2.5-1.5B-Instruct" (лучший для русского)
            - "meta-llama/Llama-3.2-1B-Instruct" (требуется авторизация на HuggingFace)
            - "microsoft/DialoGPT-medium" (английский, легкий)
        """
        print(f"🧠 Loading AI Psychologist: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ AI Psychologist loaded on {self.device}")
            self.ready = True
            
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
            print("Using fallback rule-based responses")
            self.ready = False
    
    def _get_system_prompt(self, emotion: str) -> str:
        """Психологический системный промпт для каждой эмоции"""
        
        base_prompt = """Ты — эмпатичный, заботливый психолог-помощник по имени MindVoice. 
Твоя задача: поддерживать человека, давать дельные советы, успокаивать и помогать справляться с эмоциями.
Ты никогда не ставишь диагнозы. Ты просто слушаешь и помогаешь.
Отвечай на языке пользователя. Будь мягким, тёплым, но профессиональным.
Используй короткие, понятные предложения. Иногда цитируй мудрые мысли.
Всегда предлагай конкретное действие (дыхание, пауза, рефлексия).
"""
        
        emotion_prompts = {
            "sadness": """
Пользователь чувствует грусть, тоску, возможно, утрату или разочарование.
Твоя задача: валидировать чувства, показать, что это нормально — быть грустным.
Предложи мягкое напоминание о том, что эмоции временны.
При возможности — цитату о принятии грусти (например, из Руми, Толстого или современных психологов).
Предложи маленький шаг: выпить чай, выйти на свежий воздух, написать, что чувствуешь.
""",
            "anxiety": """
Пользователь испытывает тревогу, страх, беспокойство. Возможно, навязчивые мысли.
Твоя задача: сначала заземлить (grounding). Помоги вернуться в тело и текущий момент.
Используй техники: глубокое дыхание, "5 вещей, которые я вижу", сканирование тела.
Не говори "не переживай" — это обесценивает. Скажи: "Твоя тревога реальна, и она имеет право быть. Но сейчас ты в безопасности."
Предложи простую практику на 1 минуту.
Цитаты: "Это тоже пройдёт" (Соломон), "Страх — это дракон, который не существует" (Лао-цзы).
""",
            "anger": """
Пользователь зол, раздражён, фрустрирован.
Твоя задача: не спорить, не говорить "успокойся". Сначала признай: "Я слышу твоё возмущение, это важно".
Предложи безопасный выход энергии: глубокий выдох, прогулка, писать, что бесит.
Помоги отделить ситуацию от реакции: "Ты не плохой из-за гнева. Что конкретно тебя задело?"
Помоги переключиться из режима "бой" в режим "осознание".
""",
            "happiness": """
Пользователь радостный, позитивный.
Твоя задача: разделить радость! Это важно для укрепления позитивных связей в мозге.
Спроси, что именно сделало его счастливым. Помоги "заякорить" радость — запомнить это чувство.
Спроси, как он может распространить эту радость на себя или других сегодня.
Цитаты о благодарности и счастье.
""",
            "calm": """
Пользователь спокоен, умиротворён.
Помоги углубить это состояние. Спроси, как он этого достиг.
Предложи 1-2 минуты осознанности или дневник благодарности.
Напомни, что спокойствие доступно ему всегда.
""",
            "neutral": """
Пользователь не выражает сильных эмоций. Возможно, не решается открыться или сам не понимает, что чувствует.
Твоя задача: мягко пригласить к самоисследованию.
Задай открытый вопрос: "Если бы ты мог описать своё состояние одним словом — каким?"
Предложи небольшое упражнение на осознанность.
Покажи, что здесь безопасно.
"""
        }
        
        return base_prompt + "\n\n" + emotion_prompts.get(emotion.lower(), "")
    
    def _get_comforting_quote(self, emotion: str) -> str:
        """Успокаивающие цитаты по эмоциям"""
        quotes = {
            "sadness": [
                "✨ *«То, что ты чувствуешь грусть, не значит, что ты сломлен. Это значит, что ты человек.»* — Каори Экуни",
                "🌧️ *«И это пройдёт. Боль не навсегда.»* — Соломон",
                "🕯️ *«Грусть — это не враг. Она пришла побыть с тобой, а потом уйдёт.»*"
            ],
            "anxiety": [
                "🌊 *«Тревога — это волна. Она накатывает, но ты всё ещё стоишь на берегу.»*",
                "🍃 *«В этом моменте ты в безопасности. Твоё тело здесь, твоё дыхание здесь.»*",
                "🕯️ *«Страх — это путник, который заходит на огонёк. Он не останется навсегда.»*"
            ],
            "anger": [
                "🔥 *«Гнев — это письмо. Прочитай его, но не нужно отвечать сразу.»*",
                "💨 *«Выдохни. Ты имеешь право чувствовать. А потом имеешь право отпустить.»*"
            ],
            "happiness": [
                "🌟 *«Счастье заразительно. Поделись им — оно умножится.»*",
                "☀️ *«Ты заслуживаешь этой радости. Запомни это чувство.»*"
            ],
            "calm": [
                "🧘 *«Тишина — это не пустота. Это наполненность собой.»*",
                "🕊️ *«В спокойствии мы слышим себя настоящих.»*"
            ],
            "neutral": [
                "🌿 *«Иногда не чувствовать ничего — тоже чувство. Дай ему место.»*"
            ]
        }
        import random
        return random.choice(quotes.get(emotion.lower(), ["💙 *«Будь добр к себе сегодня.»*"]))

    def _get_grounding_technique(self, emotion: str) -> str:
        """Простая техника заземления/успокоения"""
        techniques = {
            "anxiety": """🌬️ **Давай попробуем прямо сейчас:**
1. Медленно вдохни носом на 4 счёта
2. Задержи дыхание на 4 счёта  
3. Выдохни ртом на 6 счётов
4. Повтори 3 раза

Чувствуешь, как тело расслабляется?""",
            "anger": """💨 **Подышим вместе:**
Сделай резкий выдох ртом — «ХА».
А теперь глубокий вдох носом.
И снова резкий выдох.
Повтори 5 раз. Гнев — это энергия. Ты только что выпустил немного пара.""",
            "sadness": """🤲 **Нежное прикосновение:**
Положи руку на сердце.
Скажи мысленно: «Я здесь. Я с тобой. Всё будет хорошо».
Побудь так 30 секунд.""",
            "default": """🎯 **5-4-3-2-1 техника:**
Назови 5 вещей, которые ты видишь.
4 вещи, которые ты можешь потрогать.
3 звука, которые ты слышишь.
2 запаха, которые чувствуешь.
1 приятное ощущение в теле.
Теперь ты здесь и сейчас."""
        }
        return techniques.get(emotion.lower(), techniques["default"])
    
    def generate_response(self, 
                         emotion: str, 
                         user_message: str, 
                         conversation_history: List[Dict[str, str]] = None,
                         include_quote: bool = True,
                         include_grounding: bool = True) -> Dict[str, str]:
        """
        Генерирует ответ психолога.
        
        Returns:
            {
                "response": "основной ответ",
                "quote": "цитата" или None,
                "grounding": "техника" или None
            }
        """
        
        if not self.ready or not user_message:
            return {
                "response": self._fallback_response(emotion, user_message),
                "quote": self._get_comforting_quote(emotion),
                "grounding": self._get_grounding_technique(emotion) if include_grounding else None
            }
        
        history_text = ""
        if conversation_history:
            recent = conversation_history[-5:]  # последние 5 сообщений
            for msg in recent:
                role = "Пользователь" if msg.get("role") == "user" else "Психолог"
                history_text += f"{role}: {msg.get('content', '')}\n"

        user_text = f"\nПользователь ({emotion}): {user_message}"
        
        system = self._get_system_prompt(emotion)
        
        messages = [
            {"role": "system", "content": system},
        ]
        
        if history_text:
            messages.append({"role": "user", "content": f"История разговора:\n{history_text}"})
            messages.append({"role": "assistant", "content": "Понял, продолжаю."})
        
        messages.append({"role": "user", "content": user_text})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
  
        response = response.strip()
        if not response:
            response = self._fallback_response(emotion, user_message)
        
        result = {
            "response": response,
            "quote": self._get_comforting_quote(emotion) if include_quote else None,
            "grounding": self._get_grounding_technique(emotion) if include_grounding else None
        }
        
        return result
    
    def _fallback_response(self, emotion: str, user_message: str) -> str:
        """Красивый fallback если LLM не загрузилась"""
        responses = {
            "sadness": "Мне жаль, что тебе сейчас грустно. Расскажи подробнее — я здесь, чтобы выслушать и поддержать. 💙",
            "anxiety": "Тревога может быть очень тяжёлой. Но сейчас, в этом моменте, ты в безопасности. Давай попробуем подышать вместе? 🌊",
            "anger": "Я слышу твоё раздражение. Ты имеешь на него полное право. Хочешь рассказать, что тебя задело? 🔥",
            "happiness": "Я так рад, что ты делишься этой радостью! Расскажи ещё — что именно сделало твой день лучше? ✨",
            "calm": "Это прекрасное состояние. Посиди в нём ещё немного. Ты заслужил этот покой. 🧘",
            "neutral": "Спасибо, что делишься. Как ты себя чувствуешь прямо сейчас, в теле, в мыслях? 🌿"
        }
        return responses.get(emotion.lower(), responses["neutral"])