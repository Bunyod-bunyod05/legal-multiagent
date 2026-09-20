# Multi-Agent Legal Analyst — Fuqarolik kodeksi

O'zbekiston Respublikasining **fuqarolik huquqi** va **fuqarolik protsessual
huquqi** bo'yicha ko'p-agentli yordamchi. Boshqa huquq sohalari (jinoyat,
ma'muriy, mehnat va h.k.) doirasidan tashqari savollarga javob bermaydi.

Supervisor + Retriever + Web (Tavily) + Code (penya/jarima hisoblovchi)
agentlar Fuqarolik kodeksining 1- va 2-qismi (~1200 modda) ustida ishlaydi.

## Nima ishlaydi

- **💬 Suhbat** — chat, thinking-status ("Yo'l tanlanmoqda…", "Moddalar
  solishtirilmoqda…") jonli yangilanadi
- **📄 Shartnoma tahlili** — PDF/matn yuklaymiz, FK moddalariga havola bilan
  tahlil: turi, tomonlar, muhim shartlar, xatarli bandlar, tavsiyalar
- **🧮 Penya kalkulyatori** — summa × kunlik % × kunlar, FK 324–333 havolalari
- **📋 Tarix** — brauzer localStorage'ida, foydalanuvchining o'z qurilmasida
  qoladi (Streamlit bilan boshqa foydalanuvchilarga uzatilmaydi)
- **🔧 Admin ko'rinish** — `?admin=<ADMIN_TOKEN>` bilan URL ochilsa, oraliq
  bosqichlar (supervisor qarori, topilgan moddalar, kod natijasi) faqat
  admin'ga ko'rinadi

## LLM va embeddings

| Komponent | Nima | Nega |
|---|---|---|
| LLM | **Groq** (Llama 3.3 70B) → OpenRouter fallback | Groq bepul va barqaror; OpenRouter :free tier tez-tez pullikka o'tib qoladi |
| Embeddings | **fastembed** (`paraphrase-multilingual-MiniLM-L12-v2`) | Lokal ONNX, kalit yo'q, ko'p tilli (o'zbek/rus/ingliz) |
| Vektor store | **Qdrant Cloud** (persistent) yoki `:memory:` fallback | Cold-boot'da qayta embed qilmaslik uchun |

## Deploy qadamlari

### 1. GitHub va Streamlit Cloud

1. Bu repo'ni fork/clone qiling
2. https://share.streamlit.io → GitHub bilan kiring → **New app** → repo → `streamlit_app.py`
3. Deploy tugmasini bosing

### 2. Secrets (Manage app → Settings → Secrets)

**Majburiy:**
```toml
GROQ_API_KEY = "gsk_..."     # https://console.groq.com/keys — bepul, 30 sekund
```

**Ixtiyoriy:**
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."   # Groq quotasi tugasa fallback sifatida
TAVILY_API_KEY = "tvly-..."           # web-search agent uchun
QDRANT_URL = "https://xxxx.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
ADMIN_TOKEN = "<uzun tasodifiy satr>" # ?admin=<TOKEN> orqali debug ko'rish
GROQ_MODEL = "llama-3.3-70b-versatile" # boshqa Groq modeli kerak bo'lsa
OPENROUTER_MODEL = "..."               # vergul bilan ajratilgan fallback list
LLM_TIMEOUT = "20"                     # sekundlarda, model osilib qolsa fallback
```

### 3. Bir marta ingest (Qdrant Cloud bilan)

Har cold-boot'da 1200 moddani qayta embed qilmaslik uchun bir marta ishga
tushiring:

```bash
set QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
set QDRANT_API_KEY=...
python ingest.py --recreate
```

Undan keyin ilova har uyg'onganda faqat kelgan **savolni** embed qiladi —
kolleksiya Qdrant Cloud'da doimiy saqlanadi. 12 soatdan keyin uxlab qolsa
ham, keyingi kirishda kodeks qayta yuklanmaydi.

## Fayllar

- `streamlit_app.py` — asosiy ilova (graf + UI + LLM chain + fallback)
- `civil_code.py` — PDF'ni moddalarga bo'lish, fastembed bilan embed
- `ingest.py` — bir martalik Qdrant Cloud'ga yuklash skripti
- `civil_code.pdf` / `civil_code_2.pdf` — FK 1- va 2-qism
- `requirements.txt`

## Moddalarni ajratish

Kodeksda tahrirlangan moddalar yuqori indeks bilan yuriladi: 26¹, 173⁷, 358¹
va h.k. (jami 17 ta). Oddiy PDF matn olinsa ular "261", "1737" ga tekislanib,
haqiqiy 261-modda bilan bir yorliq ostiga tushib qoladi. Shuning uchun
`civil_code.py` PyMuPDF span-flag orqali superscript'ni ushlab, har moddaga
`26-1` ko'rinishidagi normal kalit beradi. Foydalanuvchi `26¹`, `26-1` yoki
`261` deb yozsa ham to'g'ri modda topiladi.

## LLM fallback zanjiri

`ResilientLLM` klassi so'rovni birinchi provider'ga yuboradi (Groq); 404 /
unavailable / rate limit / timeout / 5xx qaytsa, avtomatik keyingi model'ga
o'tadi. Kod ichida `OPENROUTER_MODELS` — vergul bilan ajratilgan slug ro'yxati.
Har biriga `LLM_TIMEOUT` sekund vaqt beriladi; osilgan model butun so'rovni
bloklab qolmaydi.

## Scope guard

Har savol avval `in_scope()` klassifikatoridan o'tadi. Fuqarolik/fuqarolik
protsessual bo'lmasa, agentlar umuman ishga tushmaydi va muloyim rad javob
qaytariladi. Bu ham quota tejaydi, ham foydalanuvchini shu doirada saqlaydi.

## Ogohlantirish

⚠️ Bu vosita huquqiy **ma'lumot** beradi, yuridik **maslahat emas**. Muhim
qarorlar uchun advokatga murojaat qiling.
