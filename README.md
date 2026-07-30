# Multi-Agent Legal Analyst — Fuqarolik kodeksi

Supervisor + Retriever + Web (Tavily) + Code (penya/jarima hisoblovchi) agentlar.

Barcha Gemini chaqiruvlari (LLM + embeddings)  LiteLLM proksi
orqali o'tadi (`https://saidazam-litellm-proxy.hf.space`), haqiqiy Google
serverlariga to'g'ridan-to'g'ri so'rov ketmaydi.

## Nega Streamlit Community Cloud (Hugging Face Spaces emas)

Hugging Face yaqinda siyosatini o'zgartirdi: Gradio/Docker Spaces endi
PRO (pullik) obuna talab qiladi. Streamlit Community Cloud esa hali ham
to'liq bepul (GitHub orqali ulanadi, cheklovlar: ~1GB RAM, 12 soat
faolsizlikdan keyin "uxlaydi" — keyingi so'rovda avtomatik uyg'onadi).

## Deploy qadamlari

1. GitHub'da yangi **public** repo yarating (masalan `legal-multiagent`)
2. Shu papkadagi barcha fayllarni (`streamlit_app.py`, `civil_code.pdf`,
   `requirements.txt`) o'sha repo'ga yuklang (push qiling)
3. https://share.streamlit.io ga kiring (GitHub hisobingiz bilan)
4. **"New app"** -> repo'ni tanlang -> **Main file path:** `streamlit_app.py`
5. **"Advanced settings" -> "Secrets"** bo'limiga qo'shing (TOML format):
   ```
   GEMINI_API_KEY = "sizning-kalitingiz"
   TAVILY_API_KEY = "sizning-kalitingiz"
   ```
6. **Deploy** tugmasini bosing, 2-5 daqiqa kuting

## Fayllar

- `streamlit_app.py` — asosiy ilova (graf + UI) — **shu fayl ishlatiladi**
- `civil_code.py` — PDF'ni moddalarga bo'lish va embedding (ilova ham,
  `ingest.py` ham shundan foydalanadi)
- `ingest.py` — bir martalik skript: kodeksni Qdrant Cloud'ga yuklaydi
- `app.py` — Gradio versiyasi (HF Spaces pullik bo'lib qolgani uchun
  hozircha ishlatilmaydi, keyinchalik kerak bo'lsa saqlanmoqda)
- `civil_code.pdf` — Fuqarolik kodeksi, **1-qism** (386 modda)
- `civil_code_2.pdf` — Fuqarolik kodeksi, **2-qism** (811 modda)
- `requirements.txt`

## Moddalarni ajratish haqida

Kodeksda tahrirlar bilan kiritilgan moddalar yuqori indeks bilan yuriladi:
26¹, 173⁷, 358¹, 1140¹ va hokazo (jami 17 ta). PDF'dan oddiy matn olinsa
ular "261", "1737", "3581" ga tekislanadi — natijada 26¹ haqiqiy 261-modda
(neustoyka shakllari) bilan bir yorliq ostiga tushib qoladi. Shuning uchun
`civil_code.py` matnni span-ma-span o'qib, PyMuPDF'ning superscript bitidan
(`span["flags"] & 1`) foydalanadi va har moddaga `26-1` ko'rinishidagi
normal kalit beradi. Foydalanuvchi `26¹`, `26-1` yoki `261` deb yozsa ham
to'g'ri modda topiladi.

## Sovuq boshlanish va Qdrant

Kalitlar berilmasa ilova butun kodeksni har uyg'onishda xotiraga embed
qiladi (~1200 modda, ~75 sekund). Embedding paketlari parallel yuboriladi:
Gemini'ning `BatchEmbedContents` limiti 100, langchain'ning `chunk_size`
defaulti esa 1000 — u shu hajmda 400 xatosi beradi, shuning uchun
`EMBED_BATCH` aniq belgilangan.

Buni butunlay yo'q qilish uchun boshqariladigan Qdrant ulanadi:

```
python ingest.py            # bir marta, GEMINI_API_KEY + QDRANT_URL bilan
```

so'ng Secrets'ga qo'shiladi:

```
QDRANT_URL = "https://xxxx.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
```

Shundan keyin ilova korpusni embed qilmaydi, faqat savolni embed qiladi.
