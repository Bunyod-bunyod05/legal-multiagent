"""
Multi-Agent Legal Analyst — Streamlit app.

Fuqarolik huquqi va Fuqarolik protsessual huquqi bo'yicha ko'p-agentli tizim.

LLM         : OpenRouter (bepul, OpenAI-mos API).
Embeddings  : fastembed (lokal ONNX, quota yo'q).
Vektor store: Qdrant Cloud (bir marta ingest → hech qachon qayta embed qilinmaydi)
              yoki :memory: (Qdrant ulanmasa, cold-boot'da lokal indeks).
Tarix       : brauzer localStorage'ida — foydalanuvchining o'z telefonida qoladi.
"""
import io
import json
import os
import re
import contextlib
import threading
import itertools
from datetime import datetime
from typing import TypedDict, List, Optional

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient, models
import fitz

from civil_code import (
    COLLECTION_NAME,
    EMBED_MODEL,
    VECTOR_SIZE,
    article_keys_in,
    embed_corpus,
    load_articles,
)

# ---------------------------------------------------------------------------
# set_page_config must be the very first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Legal Analyst", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _get_secret(name: str) -> str:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return ""


# LLM provider config. Priority: Groq (fastest free tier) → OpenRouter → error.
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GROQ_MODEL = _get_secret("GROQ_MODEL") or "llama-3.3-70b-versatile"
GROQ_BASE = "https://api.groq.com/openai/v1"

OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY")
# Comma-separated fallback list. OpenRouter frequently demotes ":free" slugs,
# so we try one, catch the 404, and roll to the next.
OPENROUTER_MODELS = [
    m.strip() for m in (
        _get_secret("OPENROUTER_MODEL")
        or "deepseek/deepseek-chat-v3.1:free,google/gemini-2.0-flash-exp:free,"
           "qwen/qwen-2.5-72b-instruct:free,mistralai/mistral-small-3.2-24b-instruct:free"
    ).split(",") if m.strip()
]
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

QDRANT_URL = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")

# Admin (debug) mode — only when ?admin=<TOKEN> matches the secret.
ADMIN_TOKEN = _get_secret("ADMIN_TOKEN")
try:
    _q = st.query_params
    ADMIN_MODE = bool(ADMIN_TOKEN) and _q.get("admin") == ADMIN_TOKEN
except Exception:
    ADMIN_MODE = False

if not (GROQ_API_KEY or OPENROUTER_API_KEY):
    st.error(
        "GROQ_API_KEY yoki OPENROUTER_API_KEY topilmadi. "
        "Bepul Groq kaliti: https://console.groq.com/keys — Streamlit Cloud "
        "Settings → Secrets bo'limiga qo'shing."
    )
    st.stop()


class ResilientLLM:
    """LLM wrapper: prefer Groq; on 404/rate-limit fall through OpenRouter list."""

    def __init__(self):
        self._llms = self._build_chain()

    def _build_chain(self):
        chain = []
        if GROQ_API_KEY:
            chain.append(("groq:" + GROQ_MODEL, ChatOpenAI(
                api_key=GROQ_API_KEY, base_url=GROQ_BASE,
                model=GROQ_MODEL, temperature=0,
            )))
        if OPENROUTER_API_KEY:
            for m in OPENROUTER_MODELS:
                chain.append(("openrouter:" + m, ChatOpenAI(
                    api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE,
                    model=m, temperature=0,
                    default_headers={
                        "HTTP-Referer": "https://legal-multiagent.streamlit.app",
                        "X-Title": "Multi-Agent Legal Analyst",
                    },
                )))
        return chain

    def invoke(self, *args, **kwargs):
        last_err = None
        for tag, llm in self._llms:
            try:
                return llm.invoke(*args, **kwargs)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # 404/unavailable/rate limit — try next model in chain.
                if any(k in msg for k in ("404", "not found", "unavailable", "rate", "429")):
                    continue
                raise
        raise last_err if last_err else RuntimeError("Barcha LLM providerlar javob bermadi")


@st.cache_resource(show_spinner=False)
def get_llms_and_embeddings():
    llm = ResilientLLM()
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return llm, llm, embeddings


# ---------------------------------------------------------------------------
# Retrieval — Qdrant Cloud persistent by default; in-memory as fallback.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fuqarolik kodeksi tayyorlanmoqda…")
def get_retriever():
    _, _, embeddings = get_llms_and_embeddings()
    articles = load_articles()
    by_key = {a["key"]: a for a in articles}
    search_kwargs = {"k": 6}

    if QDRANT_URL:
        # Persistent: kolleksiya ingest.py orqali BIR MARTA yuklanadi.
        # Ilova har uyg'onganda faqat so'rovni embed qiladi.
        store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
        return store.as_retriever(search_kwargs=search_kwargs), by_key, len(articles)

    # Fallback (Qdrant sozlanmagan): xotirada indekslash. Cold boot sekinroq.
    vectors = embed_corpus([a["text"] for a in articles])
    client = QdrantClient(location=":memory:")
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=vector,
                payload={
                    "page_content": a["text"],
                    "metadata": {"article": a["article"], "key": a["key"], "part": a["part"]},
                },
            )
            for i, (a, vector) in enumerate(zip(articles, vectors))
        ],
    )
    store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    return store.as_retriever(search_kwargs=search_kwargs), by_key, len(articles)


# ---------------------------------------------------------------------------
# Scope guard — accept ONLY civil law and civil procedural law
# ---------------------------------------------------------------------------
SCOPE_REFUSAL = (
    "Kechirasiz, men faqat **fuqarolik huquqi** va **fuqarolik protsessual huquqi** "
    "bo'yicha savollarga javob beraman. Iltimos, savolingizni shu mavzu doirasida qayta yozing."
)


def in_scope(question: str) -> bool:
    llm, _, _ = get_llms_and_embeddings()
    prompt = f"""Sen huquqiy mavzu tasniflovchisan. Savol O'zbekiston Respublikasi
FUQAROLIK HUQUQI yoki FUQAROLIK PROTSESSUAL HUQUQI doirasiga kiradimi?

Fuqarolik huquqi: shartnomalar, majburiyatlar, mulk huquqi, meros, ziyon qoplash,
penya/neustoyka, yuridik shaxslar, bitimlar, vakolat, kafolat.
Fuqarolik protsessual huquqi: sud tartibi, da'vo arizasi, isbot, sud qarorlari,
apellyatsiya, kassatsiya, ijro ishlari.

QATIY: jinoyat, ma'muriy, soliq, mehnat, oila, konstitutsiyaviy huquq VA umumiy
suhbat/texnika/siyosat — SCOPE'DAN TASHQARI.

Savol: {question}
Faqat bitta so'z: YES yoki NO."""
    raw = llm.invoke(prompt).content.strip().upper()
    return raw.startswith("YES")


# ---------------------------------------------------------------------------
# Agent state + nodes
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    plan: str
    documents: List[str]
    code_result: Optional[str]
    answer: str
    steps: List[str]
    citations: List[str]


class Route(BaseModel):
    next: str = Field(description="'retriever', 'web', 'code', yoki 'finish'")


def supervisor(state):
    llm_flash, _, _ = get_llms_and_embeddings()
    prompt = f"""Savol: {state['question']}
Bajarilgan qadamlar: {state['steps']}
Yig'ilgan ma'lumot: hujjatlar={bool(state['documents'])}, kod={state['code_result']}

Keyingi agent (bir so'z):
- retriever: kodeks moddasi haqida
- code: hisob-kitob kerak
- web: kodeksda yo'q, internetdan
- finish: yetarli ma'lumot bor

Bir marta ishlatilgan agentni qaytadan tanlama — 'finish' de.
Faqat bitta so'z: retriever/code/web/finish."""
    raw = llm_flash.invoke(prompt).content.strip().lower()
    plan = "finish"
    for keyword in ("retriever", "code", "web", "finish"):
        if keyword in raw:
            plan = keyword
            break
    return {"plan": plan, "steps": state["steps"] + [f"supervisor→{plan}"]}


def route_after_supervisor(state):
    return state["plan"]


def run_sandboxed_code(code_str: str, timeout_sec: int = 5) -> str:
    allowed_builtins = {
        "print": print, "range": range, "len": len, "round": round,
        "abs": abs, "min": min, "max": max, "sum": sum,
        "int": int, "float": float, "str": str,
    }
    safe_globals = {"__builtins__": allowed_builtins}
    buf = io.StringIO()
    box = {}

    def _runner():
        try:
            with contextlib.redirect_stdout(buf):
                exec(code_str, safe_globals, {})
        except Exception as e:
            box["error"] = f"XATOLIK: {type(e).__name__}: {e}"

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        return "XATOLIK: Vaqt limitidan oshdi"
    if "error" in box:
        return box["error"]
    return buf.getvalue().strip()


def code_agent(state):
    _, llm_lite, _ = get_llms_and_embeddings()
    prompt = f"""Quyidagi savol uchun Python kodi yoz — print() bilan natijani chiqar.
Faqat oddiy arifmetika (import kerak emas).

Savol: {state['question']}

Agar summalar/foizlar/kunlar aniq bo'lmasa, FK 326-330-moddalaridagi odatiy
penya mantig'ini (kunlik foiz × kechikkan kunlar × summa) ishlat.

Faqat Python kodini qaytar."""
    raw_code = llm_lite.invoke(prompt).content.replace("```python", "").replace("```", "").strip()
    result = run_sandboxed_code(raw_code)
    return {"code_result": result, "steps": state["steps"] + ["code"]}


def retriever_agent(state):
    retriever, by_key, n_articles = get_retriever()
    texts, cited = [], []
    for key in article_keys_in(state["question"]):
        art = by_key.get(key)
        if art and art["article"] not in cited:
            texts.append(f"[{art['article']}] {art['text']}")
            cited.append(art["article"])
    for d in retriever.invoke(state["question"]):
        label = d.metadata.get("article")
        if label in cited:
            continue
        texts.append(f"[{label}] {d.page_content}")
        cited.append(label)
    return {
        "documents": state["documents"] + texts,
        "citations": state["citations"] + cited,
        "steps": state["steps"] + [f"retriever({n_articles} modda, {len(cited)} topildi)"],
    }


def web_agent(state):
    cited = []
    try:
        tavily = TavilySearchResults(max_results=3)
        hits = tavily.invoke({"query": state["question"]})
        texts = [h["content"] for h in hits if isinstance(h, dict) and "content" in h]
        cited = [h["url"] for h in hits if isinstance(h, dict) and h.get("url")]
        step = "web"
    except Exception as e:
        texts = []
        step = f"web(xato: {type(e).__name__})"
    return {
        "documents": state["documents"] + texts,
        "citations": state["citations"] + cited,
        "steps": state["steps"] + [step],
    }


def generate(state):
    _, llm_lite, _ = get_llms_and_embeddings()
    context = "\n\n".join(state["documents"]) if state["documents"] else "(hujjat topilmadi)"
    code_part = f"\nHisoblash natijasi: {state['code_result']}" if state["code_result"] else ""
    prompt = f"""Sen O'zbekiston fuqarolik huquqi va fuqarolik protsessual huquqi
bo'yicha yordamchisan. FAQAT quyidagi kontekst asosida javob ber.

Savol: {state['question']}

Kontekst:
{context}
{code_part}

Aniq, qisqa, tartibli javob yoz. Har bir tezisda modda raqamiga havola qil.
Hisoblash natijasi bo'lsa, uni javobga aniq kiritib ko'rsat."""
    answer = llm_lite.invoke(prompt).content
    return {"answer": answer, "steps": state["steps"] + ["generate"]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_agent)
    g.add_node("web", web_agent)
    g.add_node("code", code_agent)
    g.add_node("generate", generate)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_after_supervisor, {
        "retriever": "retriever", "web": "web", "code": "code", "finish": "generate",
    })
    g.add_edge("retriever", "supervisor")
    g.add_edge("web", "supervisor")
    g.add_edge("code", "supervisor")
    g.add_edge("generate", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Streaming status — Claude-like rotating phrases
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "supervisor": "🧭 Yo'l tanlanmoqda",
    "retriever": "📚 Fuqarolik kodeksi moddalari izlanmoqda",
    "web": "🌐 Internet manbalari o'rganilmoqda",
    "code": "🧮 Hisob-kitob qilinmoqda",
    "generate": "✍️ Javob shakllantirilmoqda",
}
THINKING_PHRASES = itertools.cycle([
    "yo'l tekshirilmoqda",
    "moddalar solishtirilmoqda",
    "uch tomondan tekshirilmoqda",
    "mantiq tuzilmoqda",
    "iqtiboslar tekshirilmoqda",
    "javob aniqlashtirilmoqda",
])


def run_graph_streaming(question: str):
    initial = {
        "question": question, "plan": "", "documents": [],
        "code_result": None, "answer": "", "steps": [], "citations": [],
    }
    accumulated = dict(initial)
    with st.status("🧠 Fikrlanmoqda…", expanded=ADMIN_MODE) as status:
        for chunk in build_graph().stream(initial, {"recursion_limit": 15}):
            for node_name, node_update in chunk.items():
                for k, v in (node_update or {}).items():
                    accumulated[k] = v
                label = NODE_LABELS.get(node_name, node_name.capitalize())
                status.update(label=f"{label} · _{next(THINKING_PHRASES)}_…")
                if ADMIN_MODE:
                    st.markdown(f"**→ {node_name}**")
                    last_step = (node_update or {}).get("steps")
                    if last_step:
                        st.caption(f"step: `{last_step[-1] if isinstance(last_step, list) else last_step}`")
                    if node_name == "supervisor":
                        st.caption(f"plan: `{accumulated.get('plan')}`")
                    if node_name == "retriever":
                        cites = (node_update or {}).get("citations") or []
                        if cites:
                            st.caption("topildi: " + ", ".join(cites[:8]))
                    if node_name == "code":
                        st.code(str(accumulated.get("code_result") or ""), language="text")
        status.update(label="✅ Tayyor", state="complete", expanded=ADMIN_MODE)
    return accumulated


# ---------------------------------------------------------------------------
# localStorage-backed history (persists on user's own device)
# ---------------------------------------------------------------------------
try:
    from streamlit_local_storage import LocalStorage
    _LS = LocalStorage()
    _LS_OK = True
except Exception:
    _LS = None
    _LS_OK = False

_LS_KEY = "legal_chat_history_v1"


def load_history() -> list:
    if not _LS_OK:
        return st.session_state.get("history", [])
    raw = _LS.getItem(_LS_KEY)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def save_history(history: list) -> None:
    if not _LS_OK:
        st.session_state["history"] = history
        return
    try:
        _LS.setItem(_LS_KEY, json.dumps(history, ensure_ascii=False))
    except Exception:
        pass


def clear_history() -> None:
    if _LS_OK:
        try:
            _LS.deleteItem(_LS_KEY)
        except Exception:
            pass
    st.session_state["history"] = []


if "history" not in st.session_state:
    st.session_state.history = load_history()


# ---------------------------------------------------------------------------
# Extra tools: contract analysis, penya calculator
# ---------------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def analyze_contract(text: str) -> str:
    """Send extracted contract text to LLM with a civil-code-aware prompt."""
    llm, _, _ = get_llms_and_embeddings()
    text = text.strip()
    if len(text) > 15000:
        text = text[:15000] + "\n\n[...matn qisqartirildi...]"
    prompt = f"""Sen O'zbekiston fuqarolik huquqi bo'yicha shartnoma tahlilchisi.
Quyidagi shartnoma matnini tahlil qil. Fuqarolik kodeksining tegishli
moddalariga havola qilib, quyidagi tuzilmada javob yoz:

1. **Shartnoma turi** (masalan: xarid-sotuv, ijara, pudrat) — FK bo'yicha qanday
   institutga tegishli, tegishli moddalar.
2. **Tomonlar** — kim va kim, huquqiy shakli.
3. **Predmet va asosiy shartlar** — qisqacha.
4. **Muhim shartlar (essentialia negotii)** — FK talab qiladigan, lekin
   shartnomada YO'Q bo'lgan shartlarni sanab bering.
5. **Xatarli qoidalar** — foydalanuvchi uchun noqulay yoki FK'ga zid bandlar.
6. **Tavsiyalar** — nimalarni qo'shish/o'zgartirish kerak.

Shartnoma matni:
{text}"""
    return llm.invoke(prompt).content


PENYA_ARTICLES_NOTE = (
    "**Tegishli moddalar:** FK 260–262 (majburiyatlarni bajarish), "
    "FK 324–333 (neustoyka/penya), FK 327 (yozma shakl), "
    "FK 333 (sudning penyani kamaytirish huquqi)."
)


def calc_penya(summa: float, daily_pct: float, days: int) -> dict:
    penya = round(summa * (daily_pct / 100.0) * days, 2)
    return {
        "summa": summa,
        "daily_pct": daily_pct,
        "days": days,
        "penya": penya,
        "total": round(summa + penya, 2),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚖️ Multi-Agent Legal Analyst")
st.caption(
    "Fuqarolik huquqi va fuqarolik protsessual huquqi bo'yicha yordamchi · "
    "Supervisor + Retriever + Web + Code agentlar · FK 1- va 2-qismi (~1200 modda)."
)
if ADMIN_MODE:
    st.info("🔧 **Admin rejim yoqilgan** — barcha oraliq bosqichlar sizga ko'rinadi.")

with st.sidebar:
    st.header("📋 Tarix")
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} ta suhbat saqlangan (telefon xotirasida).")
        for i, t in enumerate(reversed(st.session_state.history[-30:])):
            q = t.get("question", "")
            when = t.get("ts", "")
            st.markdown(f"**{len(st.session_state.history) - i}.** {q[:60]}{'…' if len(q) > 60 else ''}")
            if when:
                st.caption(when)
    else:
        st.caption("Hozircha tarix bo'sh.")
    if st.button("🗑️ Tarixni tozalash", use_container_width=True):
        clear_history()
        st.rerun()

tab_chat, tab_contract, tab_penya = st.tabs(
    ["💬 Suhbat", "📄 Shartnoma tahlili", "🧮 Penya kalkulyatori"]
)


# ---- Chat tab -------------------------------------------------------------
with tab_chat:
    def render_turn(turn):
        st.write(turn["answer"])
        if turn.get("citations"):
            st.caption("Manbalar: " + " · ".join(turn["citations"]))
        if ADMIN_MODE and turn.get("steps"):
            st.caption("Bosqichlar: " + " → ".join(turn["steps"]))

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_turn(turn)

    question = st.chat_input("Savolingizni yozing (masalan: 239-modda nima haqida?)")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            if not in_scope(question):
                st.write(SCOPE_REFUSAL)
                turn = {
                    "question": question, "answer": SCOPE_REFUSAL,
                    "steps": ["scope→refused"], "citations": [],
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            else:
                result = run_graph_streaming(question)
                turn = {
                    "question": question, "answer": result["answer"],
                    "steps": result["steps"], "citations": result.get("citations", []),
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                render_turn(turn)
        st.session_state.history.append(turn)
        save_history(st.session_state.history)


# ---- Contract analysis tab ------------------------------------------------
with tab_contract:
    st.subheader("Shartnomani tekshirish")
    st.caption(
        "PDF yoki matn shaklidagi shartnomani yuklang — tizim uni fuqarolik "
        "kodeksi doirasida tahlil qiladi va xatarli bandlarni belgilaydi."
    )
    up = st.file_uploader("Shartnoma fayli (PDF yoki .txt)", type=["pdf", "txt"])
    pasted = st.text_area("…yoki matnni bevosita joylashtiring", height=180, placeholder="Shartnoma matni…")

    if st.button("🔍 Tahlil qilish", type="primary"):
        text = ""
        try:
            if up is not None:
                data = up.read()
                if up.name.lower().endswith(".pdf"):
                    text = extract_pdf_text(data)
                else:
                    text = data.decode("utf-8", errors="ignore")
            elif pasted.strip():
                text = pasted.strip()
        except Exception as e:
            st.error(f"Faylni o'qib bo'lmadi: {e}")

        if not text.strip():
            st.warning("Iltimos, fayl yuklang yoki matn joylashtiring.")
        else:
            if len(text.strip()) < 100:
                st.warning("Matn juda qisqa. Skanerlangan PDF bo'lsa, undagi rasmlardan matn olinmaydi — matnli PDF bering.")
            else:
                with st.status("📄 Shartnoma o'qilmoqda va tahlil qilinmoqda…", expanded=ADMIN_MODE) as s:
                    if ADMIN_MODE:
                        st.caption(f"belgi soni: {len(text)}")
                    result = analyze_contract(text)
                    s.update(label="✅ Tahlil tayyor", state="complete", expanded=ADMIN_MODE)
                st.markdown(result)


# ---- Penya calculator tab -------------------------------------------------
with tab_penya:
    st.subheader("Penya (neustoyka) kalkulyatori")
    st.caption(
        "Kechiktirilgan majburiyat uchun penya summasini hisoblang. Formula: "
        "**penya = summa × kunlik foiz × kechikkan kunlar**."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        summa = st.number_input("Asosiy summa (so'm)", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.2f")
    with col2:
        daily_pct = st.number_input("Kunlik foiz (%)", min_value=0.0, value=0.1, step=0.01, format="%.4f")
    with col3:
        days = st.number_input("Kechikkan kunlar", min_value=0, value=30, step=1)

    if st.button("Hisoblash", type="primary"):
        r = calc_penya(summa, daily_pct, int(days))
        c1, c2, c3 = st.columns(3)
        c1.metric("Penya", f"{r['penya']:,.2f} so'm")
        c2.metric("Umumiy to'lov", f"{r['total']:,.2f} so'm")
        c3.metric("Kunlik penya", f"{summa * daily_pct / 100:,.2f} so'm")
        st.markdown(
            f"**Hisoblash:** {summa:,.2f} × {daily_pct}% × {days} kun = **{r['penya']:,.2f} so'm** penya. "
            f"Asosiy qarz bilan birga jami — **{r['total']:,.2f} so'm**."
        )
        st.info(PENYA_ARTICLES_NOTE)
        st.caption(
            "Eslatma: FK 333-moddasiga ko'ra sud haddan tashqari yuqori penyani "
            "asosiy majburiyat oqibatlariga mos ravishda kamaytirishga haqli."
        )


st.divider()
st.caption(
    "⚠️ Bu vosita huquqiy ma'lumot beradi, yuridik maslahat emas. "
    "Muhim qarorlar uchun advokatga murojaat qiling."
)
