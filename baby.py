import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer
import speech_recognition as sr
import pyttsx3
import json
from collections import defaultdict
import time

# ---------------------------
# 1. سیستم حافظه (خاطرات و تجربیات)
# ---------------------------
class MemorySystem:
    def __init__(self):
        self.memories = []
        self.importance = defaultdict(float)
        
    def add_memory(self, event, emotion, importance=0.5):
        memory_vec = self._text_to_vector(event)
        self.memories.append({
            "vector": memory_vec,
            "text": event,
            "emotion": emotion,
            "importance": importance
        })
        self._update_importance(event, importance)
    
    def recall(self, query, top_k=3):
        query_vec = self._text_to_vector(query)
        similarities = []
        for mem in self.memories:
            sim = np.dot(query_vec, mem["vector"]) / (np.linalg.norm(query_vec) * np.linalg.norm(mem["vector"]))
            similarities.append((sim, mem))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [mem for (sim, mem) in similarities[:top_k]]
    
    def _text_to_vector(self, text):
        words = text.lower().split()
        vec = np.zeros(100)
        for word in words:
            vec += np.random.rand(100)  # جایگزین با مدل واقعی مانند Word2Vec
        return vec / (np.linalg.norm(vec) + 1e-10)
    
    def _update_importance(self, event, delta):
        self.importance[event] += delta

# ---------------------------
# 2. یادگیری تقویتی (سیستم پاداش/تنبیه)
# ---------------------------
class ReinforcementLearner:
    def __init__(self):
        self.q_table = defaultdict(dict)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
    def update(self, state, action, reward, next_state):
        # ایجاد حالت‌ها اگر وجود نداشته باشند
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        best_next_action = max(self.q_table[next_state].values(), default=0)
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def get_best_action(self, state):
        if not self.q_table.get(state):
            return None  # یا پاسخ پیش‌فرض
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
# ---------------------------
# 3. پردازش زبان طبیعی (NLP)
# ---------------------------
class NLUProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.emotion_labels = ["شادی", "غم", "ترس", "خشم", "تعجب", "بی‌تفاوتی"]
        
    def generate_response(self, context, max_length=50):
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def detect_emotion(self, text):
        # تحلیل احساسات متن (مدل ساده)
        positive_words = ["خوب", "شاد", "عالی"]
        negative_words = ["بد", "غمگین", "ناراحت"]
        
        if any(word in text for word in positive_words):
            return "شادی"
        elif any(word in text for word in negative_words):
            return "غم"
        return "بی‌تفاوتی"

# ---------------------------
# 4. شبکه عصبی (تصمیم‌گیری شناختی)
# ---------------------------
class CognitiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)  # 6 حالت احساسی
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# ---------------------------
# 5. سیستم صوتی
# ---------------------------
class VoiceSystem:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
    
    def listen(self):
        """گوش دادن به صدای کاربر و تبدیل به متن"""
        with sr.Microphone() as source:
            print("\n🎤 در حال گوش دادن... (برای خروج بگویید 'خداحافظ')")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio, language="fa-IR")
                print(f"👤 شما: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                print("زمان شنیدن به پایان رسید")
                return None
            except Exception as e:
                print(f"خطا در تشخیص صدا: {e}")
                return None
    
    def speak(self, text):
        """پاسخ صوتی"""
        print(f"🤖: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

# ---------------------------
# 6. هسته اصلی هوش مصنوعی
# ---------------------------
class HumanLikeAI:
    def __init__(self):
        self.memory = MemorySystem()
        self.learner = ReinforcementLearner()
        self.nlu = NLUProcessor()
        self.brain = CognitiveNetwork()
        self.voice = VoiceSystem()
        
        # بارگیری دانش اولیه
        self._load_base_knowledge()
    
    def _load_base_knowledge(self):
        base_data = [
            ("سلام", "سلام! چطور می‌تونم کمک کنم؟", "شادی"),
            ("خداحافظ", "به امید دیدار!", "غم"),
            ("اسم تو چیه", "من یک هوش مصنوعی انسان‌گونه هستم", "بی‌تفاوتی")
        ]
        for text, response, emotion in base_data:
            self.memory.add_memory(text, emotion, importance=1.0)
            self.learner.q_table[text][response] = 1.0
    
    def process(self, input_text):
        # تحلیل احساسات
        emotion = self.nlu.detect_emotion(input_text)
        
        # بازیابی از حافظه
        memories = self.memory.recall(input_text)
        best_memory = memories[0] if memories else None
        
        # تولید پاسخ
        if best_memory and best_memory["importance"] > 0.7:
            response = best_memory["text"]
        else:
            response = self.nlu.generate_response(input_text)
        
        # یادگیری
        reward = 0.8 if best_memory else 0.3
        self.learner.update(input_text, response, reward, input_text)
        self.memory.add_memory(input_text, emotion)
        
        return response, emotion
    
    def run(self):
     print("====== سیستم هوش مصنوعی انسان‌گونه فعال شد ======")
     while True:
        user_input = self.voice.listen()  # حالا دیگر خطا نمی‌دهد
        
        if not user_input:
            self.voice.speak("نتونستم صدات رو بشنوم، دوباره بگو")
            continue
            
        if "خداحافظ" in user_input:
            self.voice.speak("خدانگهدار! هر وقت خواستی حرف بزنیم دوباره صدام کن.")
            break
            
        response, emotion = self.process(user_input)
        self.voice.speak(response)

# ---------------------------
# اجرای برنامه
# ---------------------------
if __name__ == "__main__":
    ai = HumanLikeAI()
    ai.run()



    "امیرنصر، بیدار شو! ادامه می‌دیم..."