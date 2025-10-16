
Baby AI is a Python-based, human-like conversational agent designed to learn and adapt in real time through memory and feedback. The project combines memory-augmented learning, simple reinforcement learning, natural language understanding, and a lightweight cognitive network to simulate a developing child’s curiosity and growth. It is implemented with widely used open-source libraries such as PyTorch and the Transformers ecosystem, enabling researchers and developers to experiment with language generation, memory retrieval, and decision-making in a compact, extensible package.

Core concept:
At its heart, Baby AI maintains a structured memory of past interactions as experiences, each annotated with an emotional tag and an importance weight. When a new user input arrives, the system recalls the most relevant memories and uses them to inform the response. If recall quality is insufficient, it falls back to a language model (GPT-2) to generate a contextually appropriate reply. This design enables the agent to simulate ongoing learning across conversations, gradually aligning its behavior with user preferences and accumulated experiences.

Memory subsystem:
The memory component encodes textual events into vector representations and supports recall by similarity to a query. Memories are stored with associated emotion labels and an importance score, which guides retrieval and future updates. This module provides a foundation for experiential learning and personalized interactions over time.

Learning and adaptation:
Baby AI implements a simple reinforcement-learning loop. The agent updates its action-value estimates (Q-values) using observed rewards and transitions, enabling it to refine its responses based on user feedback. An initial set of base memories seeds the agent’s early behavior, while continuous experiences shape evolving preferences.

Natural language processing:
A GPT-2-based language model generates fluent, context-aware responses. An emotion detector, even in a lightweight form, offers a first-pass qualitative understanding of user sentiment, enabling the system to adjust tone and content. The NLP component is designed to be modular, allowing easy replacement with more advanced models as needed.

Cognitive control:
A lightweight neural network serves as the cognitive backbone, providing structured decision-making for emotional labeling and action selection. This component can be extended to handle more nuanced affective states or to integrate with external perception modules.

Voice interface:
Optional voice I/O is supported via SpeechRecognition for speech-to-text and pyttsx3 for text-to-speech, enabling hands-free interaction. The architecture supports both audio and text-based interfaces, making Baby AI suitable for diverse use cases, from educational tools to interactive assistants.

Usage and extensibility:
The repository is organized for easy extension: modular modules, clear APIs, and documented interfaces. Developers can swap in alternative models, extend memory schemas, or integrate new reward structures. The project aims to be a friendly starting point for researchers and students exploring emergent behavior in learning agents.


# baby AI

baby AI یک هوش مصنوعی انسان‌گونه است که به صورت تعاملی از طریق حافظه و بازخورد یاد می‌گیرد. این پروژه با پایتون نوشته شده و از مدل زبان GPT-2 برای تولید پاسخ‌ها استفاده می‌کند و یک سیستم حافظه-یادگیری برای ذخیره تجربیات و بهبود پاسخ‌ها فراهم می‌کند. علاوه بر پاسخ متنی، امکان پاسخ صوتی نیز وجود دارد (در صورت فعال بودن کتابخانه‌های صوتی).

## توضیح کوتاه

baby AI تجربه یک مکالمه طبیعی را شبیه به گفتگوی با یک کودک انسان‌گونه ارائه می‌دهد: نگهداری خاطرات، یادگیری از بازخورد، و پاسخ‌گویی به کاربر با استفاده از مدل زبانی و یک شبکه تصمیم‌گیری ساده.

---

## ویژگی‌های کلیدی

- حافظه یادگیری همگرا: ذخیره خاطرات و تجربیات با مدیریت اهمیت
- یادگیری سطحی/تقویتی پایه: به‌روزرسانی پاسخ‌ها از طریق بازخورد
- پردازش زبان طبیعی: پاسخ با GPT-2 و تشخیص احساس ساده
- تصمیم‌گیری شناختی: شبکه عصبی ساده برای طبقه‌بندی حالات احساسی
- ورودی/خروجی صوتی (اختیاری): تشخیص گفتار و تبدیل پاسخ به گفتار
- رابط کاربری گفتگویی: تعامل بین صوتی و متنی

---

## معماری و ساختار پروژه

- زبان‌ها و فناوری‌ها:
  - Python
  - PyTorch (شبکه عصبی ساده)
  - transformers (GPT-2) و tokenizer
  - NumPy
  - SpeechRecognition (ورودی صوتی) و pyttsx3 (خروجی صوتی) — در صورت استفاده

- ماژول‌های اصلی (نام‌های پیشنهادی):
  - memory.py / memory_system: مدیریت خاطرات و بازخوانی
  - reinforcement.py / reinforcement_learning: یادگیری تقویتی پایه
  - nlu.py / nlu_processor: پردازش زبان طبیعی و تشخیص احساس
  - cognitive_network.py: شبکه عصبی تصمیم‌گیری
  - voice.py: ورودی/خروجی صوتی
  - ai.py / main.py: هسته یکپارچه

- ساختار دایرکتوری پیشنهادی:

- baby_ai/
├── src/
│ ├── memory.py
│ ├── reinforcement.py
│ ├── nlu.py
│ ├── cognitive_network.py
│ ├── voice.py
│ └── ai.py
├── models/
│ └── pretrained_gpt2/ (دانلود شده در صورت نیاز)
├── tests/
├── README.md
├── requirements.txt
└── LICENSE


---  

## نصب و راه‌اندازی  

1) ایجاد محیط مجازی  
- Linux/macOS:  
  - python3 -m venv venv  
  - source venv/bin/activate  
- Windows:  
  - python -m venv venv  
  - venv\Scripts\activate  

2) نصب وابستگی‌ها  
- pip install -r requirements.txt  

3) دانلود مدل‌های لازم  
- مدل GPT-2 را از طریق کتابخانه ترنسفورمرها دانلود کنید و در پوشه models/pretrained_gpt2 قرار دهید (اگر از مدل خام استفاده می‌کنید).  

4) اجرای برنامه  
- با توجه به ساختار پروژه خود:  
  - python -m baby_ai  
  - یا python src/ai.py  

5) نکات اضافی  
- اگر از ورودی صوتی استفاده می‌کنید:  
  - مطمئن شوید که کاربر می‌تواند به میکروفون دسترسی داشته باشد.  
  - کتابخانه‌های صوتی (SpeechRecognition، PyAudio یا alternatives) باید نصب شوند.  
- اگر ابزار صوتی فعال نیست، از ورودی متنی بهره ببرید.  

---  

## نحوه استفاده (نمونه‌ها)  

- اجرای مکالمه متنی ساده (فرضاً با یک ورودی از کاربر):  
  - python -m baby_ai  
  - یا python src/ai.py  

- استفاده از ورودی صوتی (در صورت پشتیبانی):  
  - اجرای اسکریپت و
