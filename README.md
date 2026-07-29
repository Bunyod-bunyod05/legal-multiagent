# Multi-Agent Legal Analyst — Fuqarolik kodeksi

Supervisor + Retriever + Web (Tavily) + Code (penya/jarima hisoblovchi) agentlar.

Barcha Gemini chaqiruvlari (LLM + embeddings) ustozning LiteLLM proksisi
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
- `app.py` — Gradio versiyasi (HF Spaces pullik bo'lib qolgani uchun
  hozircha ishlatilmaydi, keyinchalik kerak bo'lsa saqlanmoqda)
- `civil_code.pdf` — Fuqarolik kodeksi (1-qism), ishga tushganda ingest qilinadi
- `requirements.txt`
