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
st.set_page_config(
    page_title="Multi-Agent Legal Analyst",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        # Order: fastest small model first, then medium, then large as final fallback.
        or "nvidia/nemotron-nano-9b-v2:free,"
           "google/gemini-2.0-flash-exp:free,"
           "deepseek/deepseek-chat-v3.1:free,"
           "nvidia/nemotron-3-ultra-550b-a55b:free"
    ).split(",") if m.strip()
]
# Per-request timeout (seconds). Slow/hung models roll to the next in chain.
LLM_TIMEOUT = int(_get_secret("LLM_TIMEOUT") or "20")
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
        "🔑 **GROQ_API_KEY topilmadi.** Bepul kalit oling: "
        "https://console.groq.com/keys (30 soniyada, Google/GitHub bilan sign up). "
        "Keyin Streamlit Cloud → Manage app → Settings → Secrets bo'limiga "
        "`GROQ_API_KEY = \"gsk_...\"` deb qo'shing va Reboot bosing."
    )
    st.stop()

if not GROQ_API_KEY:
    st.warning(
        "⚠️ Faqat OpenRouter ishlatilmoqda — bepul modellar sekin/uzilib qoladi. "
        "Groq (https://console.groq.com/keys) qo'shsangiz sezilarli tezlashadi."
    )


class ResilientLLM:
    """LLM wrapper: prefer Groq; on 404/rate-limit fall through OpenRouter list."""

    def __init__(self):
        self._llms = self._build_chain()

    def _build_chain(self):
        chain = []
        if GROQ_API_KEY:
            chain.append(("groq:" + GROQ_MODEL, ChatOpenAI(
                api_key=GROQ_API_KEY, base_url=GROQ_BASE,
                model=GROQ_MODEL, temperature=0, timeout=LLM_TIMEOUT, max_retries=1,
            )))
        if OPENROUTER_API_KEY:
            for m in OPENROUTER_MODELS:
                chain.append(("openrouter:" + m, ChatOpenAI(
                    api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE,
                    model=m, temperature=0, timeout=LLM_TIMEOUT, max_retries=1,
                    default_headers={
                        "HTTP-Referer": "https://bperro.streamlit.app",
                        "X-Title": "Multi-Agent Legal Analyst",
                    },
                )))
        return chain

    def invoke(self, *args, **kwargs):
        errors = []
        for tag, llm in self._llms:
            try:
                return llm.invoke(*args, **kwargs)
            except Exception as e:
                errors.append(f"{tag}: {type(e).__name__}: {str(e)[:200]}")
                # Any error → try next provider. The chain is our resilience.
                continue
        raise RuntimeError(
            "Barcha LLM providerlar javob bermadi. Iltimos, biroz kutib qayta "
            "urinib ko'ring yoki GROQ_API_KEY qo'shing. Xatolar:\n- "
            + "\n- ".join(errors)
        )


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
    search_kwargs = {"k": 10}

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
    difficulty: str


class Route(BaseModel):
    next: str = Field(description="'retriever', 'web', 'code', yoki 'finish'")


def supervisor(state):
    difficulty = state.get("difficulty") or "orta"

    # Fast path: on "oson", never call an extra LLM for routing — retriever
    # once, then finish. This cuts a full LLM roundtrip and answers ~2× faster.
    if difficulty == "oson":
        if not state.get("documents"):
            return {"plan": "retriever", "steps": state["steps"] + ["supervisor→retriever (oson)"]}
        return {"plan": "finish", "steps": state["steps"] + ["supervisor→finish (oson)"]}

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


def _rewrite_query_for_retrieval(question: str, llm) -> str:
    """Turn a lay question into legal search keywords for the vector index."""
    prompt = f"""Foydalanuvchi savolini O'zbekiston Fuqarolik kodeksi bo'yicha
qidirish uchun kalit so'zlarga aylantir. Faqat qidiruv uchun kalit so'zlarni
yoz — izohsiz, sonsiz, jumlasiz. Huquqiy institut nomlari, shartnoma turlari,
kodeks tushunchalarini keltir.

Misol:
Savol: "Do'kondan kartoshka oldim, chek bermadi."
Kalit so'zlar: chakana savdo shartnomasi, kassa cheki, savdo hujjati, iste'molchi huquqlari, sotuvchi majburiyatlari

Misol:
Savol: "Uy sotayotgan edim, xaridor puldan qochyapti."
Kalit so'zlar: xarid-sotuv shartnomasi, ko'chmas mulk, majburiyatni bajarmaslik, penya, sudga da'vo

Savol: "{question}"
Kalit so'zlar:"""
    try:
        raw = llm.invoke(prompt).content.strip()
        if raw and len(raw) < 400:
            return raw
    except Exception:
        pass
    return question


def retriever_agent(state):
    retriever, by_key, n_articles = get_retriever()
    llm, _, _ = get_llms_and_embeddings()
    texts, cited = [], []

    # Exact "N-modda" lookup — always try first.
    for key in article_keys_in(state["question"]):
        art = by_key.get(key)
        if art and art["article"] not in cited:
            texts.append(f"[{art['article']}] {art['text']}")
            cited.append(art["article"])

    # Vector search over BOTH the raw question and a legal-keyword rewrite.
    # Users often ask in everyday Uzbek ("chek bermadi", "kartoshka oldim");
    # embedding those directly ranks unrelated articles because the shared
    # vocabulary with the code is thin. A rewrite into "chakana savdo
    # shartnomasi, kassa cheki, iste'molchi huquqlari" lands on the right ones.
    queries = [state["question"]]
    if (state.get("difficulty") or "orta") != "oson":
        rewritten = _rewrite_query_for_retrieval(state["question"], llm)
        if rewritten and rewritten != state["question"]:
            queries.append(rewritten)

    for q in queries:
        for d in retriever.invoke(q):
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
    difficulty = state.get("difficulty") or "orta"

    if difficulty == "oson":
        style = ("Juda qisqa, 2-4 jumla. Faqat asosiy javob, ortiqcha izohsiz. "
                 "Modda raqamini qavs ichida ber.")
    elif difficulty == "qiyin":
        style = ("Chuqur va batafsil javob yoz. Har bir tezisga modda raqamini "
                 "havola qil, doktrinal izohlar, poydevorli asoslash, muqobil "
                 "talqinlar bo'lsa keltir. Kerak bo'lsa bo'limlarga bo'l.")
    else:
        style = ("Aniq, tartibli, o'rtacha uzunlikda javob. Har bir tezisga "
                 "modda raqamini havola qil.")

    prompt = f"""Sen O'zbekiston fuqarolik huquqi va fuqarolik protsessual huquqi
bo'yicha yordamchisan. Foydalanuvchining real amaliy holatiga foydali javob
berish — asosiy vazifang.

Savol: {state['question']}

Kontekst (moddadan olib berilgan matnlar, mos yoki mos emas bo'lishi mumkin):
{context}
{code_part}

QOIDALAR:
1. Avval savolga TO'G'RIDAN-TO'G'RI javob ber — foydalanuvchi shifokorga
   emas, hayotiy muammoga yechim izlab keladi. "Kontekstda yo'q" deb qo'l
   siltama.
2. Kontekstdagi mos MODDA bo'lsa — uni tanla va raqamiga havola qilib
   izohla (masalan: "FK 434-modda bo'yicha…"). Nomos moddalarni tashlab yubor.
3. Kontekst muammoni to'liq qamramasa — O'zbekiston fuqarolik huquqidagi
   umumiy tamoyillar (masalan, chakana savdo shartnomasi, iste'molchi
   huquqlari) asosida javob ber, lekin aniq modda raqamini AYTMA agar
   ishonchli bilmasang. "FK ning tegishli moddasi bo'yicha…" deb yoz.
4. Foydalanuvchi qanday harakat qilishi mumkinligini ayt (masalan: shikoyat
   arizasi, sudga da'vo, qaysi organga murojaat).
5. Agar masala Fuqarolik kodeksidan tashqari qonunlarga (masalan,
   "Iste'molchi huquqlarini himoya qilish to'g'risida"gi qonun,
   "Kassoviy apparatlar…" qonuni) ham tegishli bo'lsa — buni oddiy
   tilda ayt, foydalanuvchi qayerga qarashi kerakligini ko'rsat.

Javob uslubi: {style}
Hisoblash natijasi bo'lsa, uni javobga aniq kiritib ko'rsat."""
    answer = llm_lite.invoke(prompt).content
    return {"answer": answer, "steps": state["steps"] + [f"generate ({difficulty})"]}


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


DIFFICULTY_PROFILES = {
    "oson":  {"recursion": 4,  "label": "🐇 Oson (tez, sodda)"},
    "orta":  {"recursion": 10, "label": "🐢 O'rta (balansli)"},
    "qiyin": {"recursion": 20, "label": "🦉 Qiyin (chuqur tahlil)"},
}


def run_graph_streaming(question: str, difficulty: str = "orta"):
    profile = DIFFICULTY_PROFILES.get(difficulty, DIFFICULTY_PROFILES["orta"])
    initial = {
        "question": question, "plan": "", "documents": [],
        "code_result": None, "answer": "", "steps": [], "citations": [],
    }
    initial["difficulty"] = difficulty
    accumulated = dict(initial)
    with st.status(f"🧠 Fikrlanmoqda… ({profile['label']})", expanded=ADMIN_MODE) as status:
        for chunk in build_graph().stream(initial, {"recursion_limit": profile["recursion"]}):
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
# Auth (Google) — Streamlit 1.42+ has built-in OIDC via st.login/st.user.
# Also, per-device fingerprint (canvas + WebGL + screen + timezone hash) so
# that a returning anonymous device is at least recognized as "the same one".
# ---------------------------------------------------------------------------
FREE_LIMIT = int(_get_secret("FREE_LIMIT") or "5")
ADMIN_EMAIL = _get_secret("ADMIN_EMAIL") or "bunyodpanjiyev48@gmail.com"
_ACCESS_CODES = {c.strip() for c in (_get_secret("ACCESS_CODES") or "").split(",") if c.strip()}
_COUNTER_KEY = "legal_request_count_v1"
_UNLOCK_KEY = "legal_unlocked_v1"

try:
    from streamlit_javascript import st_javascript
    _JS_OK = True
except Exception:
    _JS_OK = False


def get_fingerprint() -> str:
    """Canvas+WebGL+screen+timezone+userAgent → SHA-256. None on first pass."""
    if st.session_state.get("_fp"):
        return st.session_state["_fp"]
    if not _JS_OK:
        return ""
    fp = st_javascript(
        """(async () => {
            try {
              const c = document.createElement('canvas');
              const g = c.getContext('2d');
              g.textBaseline = 'top';
              g.font = "14px 'Arial'";
              g.fillStyle = '#f60';
              g.fillRect(125, 1, 62, 20);
              g.fillStyle = '#069';
              g.fillText('legal-multiagent', 2, 15);
              const gl = document.createElement('canvas').getContext('webgl');
              const glInfo = gl ? (gl.getParameter(gl.VERSION) + '|' + gl.getParameter(gl.RENDERER)) : '';
              const parts = [
                c.toDataURL(),
                glInfo,
                navigator.userAgent,
                screen.width + 'x' + screen.height + 'x' + screen.colorDepth,
                Intl.DateTimeFormat().resolvedOptions().timeZone,
                navigator.language,
                (navigator.languages || []).join(','),
                navigator.hardwareConcurrency || 0,
                navigator.platform || '',
                navigator.maxTouchPoints || 0,
              ].join('|');
              const buf = new TextEncoder().encode(parts);
              const hash = await crypto.subtle.digest('SHA-256', buf);
              return Array.from(new Uint8Array(hash))
                .map(b => b.toString(16).padStart(2, '0')).join('');
            } catch (e) { return 'err_' + (e && e.message || 'x'); }
        })()""",
        key="fp_js",
    )
    if fp and isinstance(fp, str) and len(fp) >= 32:
        st.session_state["_fp"] = fp
        return fp
    return ""


def is_google_logged_in() -> bool:
    try:
        return bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        return False


def google_auth_configured() -> bool:
    """True when Streamlit secrets carries the [auth] block for OIDC."""
    if not hasattr(st, "login"):
        return False
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def current_user_label() -> str:
    if is_google_logged_in():
        try:
            return f"{st.user.name or ''} <{st.user.email or ''}>".strip()
        except Exception:
            return "kirilgan"
    return ""


def get_usage_count() -> int:
    if not _LS_OK:
        return st.session_state.get("_req_count", 0)
    try:
        return int(_LS.getItem(_COUNTER_KEY) or "0")
    except Exception:
        return 0


def is_unlocked() -> bool:
    if not _LS_OK:
        return st.session_state.get("_unlocked", False)
    try:
        return _LS.getItem(_UNLOCK_KEY) == "true"
    except Exception:
        return False


def increment_usage() -> None:
    n = get_usage_count() + 1
    if _LS_OK:
        try:
            _LS.setItem(_COUNTER_KEY, str(n))
            return
        except Exception:
            pass
    st.session_state["_req_count"] = n


def try_unlock(code: str) -> bool:
    if code and code in _ACCESS_CODES:
        if _LS_OK:
            try:
                _LS.setItem(_UNLOCK_KEY, "true")
            except Exception:
                pass
        st.session_state["_unlocked"] = True
        return True
    return False


def _mailto(subject: str) -> str:
    from urllib.parse import quote
    return f"mailto:{ADMIN_EMAIL}?subject={quote(subject)}"


def render_paywall(context: str) -> None:
    """Show login/signup block. Caller must st.stop() after."""
    st.warning(
        f"🔒 **Bepul chegara tugadi** ({FREE_LIMIT} so'rov). Cheksiz foydalanish "
        f"uchun kiring yoki ro'yxatdan o'ting."
    )
    if google_auth_configured():
        st.markdown("#### 🔐 Google orqali kirish")
        st.caption(
            "Google akkauntingiz bilan kirasiz — parol yaratish shart emas, "
            "cheksiz so'rov beriladi."
        )
        st.button(
            "🅶 Google bilan kirish / ro'yxatdan o'tish",
            type="primary",
            on_click=lambda: st.login("google"),
            key=f"login_g_{context}",
        )
        st.markdown("---")
    else:
        st.info(
            "ℹ️ Google login hozir sozlanmagan — kirish kodi bilan davom eting "
            "yoki admin bilan bog'laning."
        )

    _subject = "Legal-Multiagent — kirish kodini so'rayman"
    st.markdown(
        f"**Yoki kirish kodi bilan:** [{ADMIN_EMAIL}]({_mailto(_subject)}) "
        "ga xat yozing, kod yuboriladi."
    )
    with st.form(f"unlock_{context}"):
        code = st.text_input("Kirish kodi", type="password", placeholder="Sizga yuborilgan kod…")
        if st.form_submit_button("🔓 Ochish"):
            if try_unlock(code.strip()):
                st.success("Kod qabul qilindi. Endi cheksiz foydalanishingiz mumkin.")
                st.rerun()
            else:
                st.error("Kod noto'g'ri yoki eskirgan.")


def rate_limit_gate(context: str) -> bool:
    """Return True if request is allowed. If not, render paywall."""
    if is_google_logged_in() or is_unlocked():
        return True
    if get_usage_count() < FREE_LIMIT:
        return True
    render_paywall(context)
    return False


# ---------------------------------------------------------------------------
# Extra tools: contract analysis, penya calculator
# ---------------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def analyze_contract(text: str) -> str:
    """Contract risk review — only risky clauses + recommendations."""
    llm, _, _ = get_llms_and_embeddings()
    text = text.strip()
    if len(text) > 15000:
        text = text[:15000] + "\n\n[...matn qisqartirildi...]"
    prompt = f"""Sen O'zbekiston fuqarolik huquqi bo'yicha shartnoma
tahlilchisisan. Quyidagi shartnomani o'qib, FAQAT ikkita bo'lim yoz — boshqa
hech narsa qo'shma (kirish, xulosa, shartnoma turi, tomonlar — bularning
hech biri kerak emas).

## MUHIM TIL QOIDALARI
- Sof, ravon o'zbek tilida yoz. Rus/ingliz kalka tarjimalarini ishlatma.
- Anglashilmovchilikka olib keladigan so'zlarni ishlatma. Masalan, "rounding"
  (qoldiqni yuqoriga tenglashtirish) ni "yuvarlash" deb tarjima qilma —
  "pul yuvish" bilan aralashib ketadi. O'rniga: "qoldiqni tenglashtirish"
  yoki "summani butunlashtirish".
- Aniqlik: har xavfli banddan keyin "chunki [aniq sabab]" deb yoz.
- Iqtibos: FK moddasi raqamini qavs ichida ko'rsat (masalan: "(FK 374-modda)").
  Modda mavjudligini bilmasang — havolani umuman berma, "FK ning tegishli
  moddasi bo'yicha" deb yoz.
- Ortiqcha jargon, takrorlash va so'z o'yinini kesib tashla.

## 1. Xatarli bandlar
Foydalanuvchi (mijoz/fuqaro) uchun noqulay yoki FK'ga zid bandlarni sanab
ber. Har biri uchun:
- **Band raqami va qisqacha mazmuni** (bir jumla)
- **Nima xavfli** (bir-ikki jumla, aniq oqibat bilan)
- **Huquqiy asos** (FK moddasi, bilsang)

Format: har band alohida qism sifatida, raqamlangan.

## 2. Tavsiyalar
Har bir xatarli band uchun aniq nima o'zgartirish yoki qo'shish kerak.
Format: raqamlangan ro'yxat, bir tavsiya bir jumla.

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
# Trigger the fingerprint computation early so it is ready by the time the
# user submits a request. Value returns on the next rerun.
_ = get_fingerprint()

# ---------------------------------------------------------------------------
# Sidebar — ChatGPT/Gemini-style history + auth block, native ← collapse arrow
# plus a custom ✕ that clicks the same button via JS in one tap.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <style>
          .close-x {
            float: right; cursor: pointer; padding: 2px 8px;
            font-size: 18px; color: #888; user-select: none;
            border-radius: 6px;
          }
          .close-x:hover { color: #000; background: #eee; }
        </style>
        <div>
          <span style='font-weight:600; font-size:18px'>📋 Tarix</span>
          <span class='close-x' title='Yopish' onclick="
            const doc = window.parent.document;
            const nodes = doc.querySelectorAll('button, [role=button]');
            for (const b of nodes) {
              const lbl = ((b.getAttribute('aria-label')||'') + ' ' +
                           (b.getAttribute('data-testid')||'')).toLowerCase();
              if (lbl.includes('collaps') || lbl.includes('close sidebar')) {
                b.click(); return;
              }
            }
          ">✕</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} ta suhbat — qurilma xotirasida.")
        for i, t in enumerate(reversed(st.session_state.history[-30:])):
            q = t.get("question", "")
            when = t.get("ts", "")
            st.markdown(f"**{len(st.session_state.history) - i}.** {q[:55]}{'…' if len(q) > 55 else ''}")
            if when:
                st.caption(when)
        if st.button("🗑️ Tarixni tozalash", use_container_width=True, key="clear_history_btn"):
            clear_history()
            st.rerun()
    else:
        st.caption("Hozircha tarix bo'sh.")

    st.markdown("---")
    if is_google_logged_in():
        st.markdown(f"👤 **{current_user_label()}**")
        st.button("Chiqish", on_click=lambda: st.logout(), use_container_width=True, key="logout_side")
    elif google_auth_configured():
        st.button(
            "🅶 Google bilan kirish",
            on_click=lambda: st.login("google"),
            use_container_width=True,
            key="login_side",
            type="primary",
        )
    else:
        st.caption("_(Google login sozlanmagan)_")

st.title("⚖️ Xush kelibsiz!")
st.caption("Sizga qanday yordam bera olaman?")

# Quota/status chip
if ADMIN_MODE:
    st.info("🔧 **Admin rejim yoqilgan** — barcha oraliq bosqichlar sizga ko'rinadi.")
elif is_google_logged_in():
    st.success(f"👤 {current_user_label()} — cheksiz kirish yoqilgan.")
elif not is_unlocked():
    _remaining = max(0, FREE_LIMIT - get_usage_count())
    if _remaining > 0:
        st.caption(f"🎁 Bepul so'rovlar qoldi: **{_remaining} / {FREE_LIMIT}**")

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

    # Difficulty selector — user picks how deep the model should think.
    _diff_col, _info_col = st.columns([2, 3])
    with _diff_col:
        difficulty = st.radio(
            "Savol qiyinligi",
            options=["oson", "orta", "qiyin"],
            format_func=lambda k: DIFFICULTY_PROFILES[k]["label"],
            index=1, horizontal=True, key="difficulty_radio",
        )
    with _info_col:
        st.caption(
            "🐇 **Oson** — 1 chaqiruv, qisqa javob (~2-5s). "
            "🐢 **O'rta** — supervisor + retriever (~10-20s). "
            "🦉 **Qiyin** — batafsil ko'p-agentli tahlil (~30-60s)."
        )

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_turn(turn)

    question = st.chat_input("Savolingizni yozing (masalan: 239-modda nima haqida?)")

    if question:
        if not rate_limit_gate("chat"):
            st.stop()
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            turn = None
            try:
                if not in_scope(question):
                    st.write(SCOPE_REFUSAL)
                    turn = {
                        "question": question, "answer": SCOPE_REFUSAL,
                        "steps": ["scope→refused"], "citations": [],
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                else:
                    result = run_graph_streaming(question, difficulty=difficulty)
                    turn = {
                        "question": question, "answer": result["answer"],
                        "steps": result["steps"], "citations": result.get("citations", []),
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "difficulty": difficulty,
                    }
                    render_turn(turn)
            except Exception as e:
                st.error(
                    f"❌ Javob olinmadi — LLM provider javob bermadi.\n\n"
                    f"**Iltimos:** biroz kutib qayta urinib ko'ring yoki Groq "
                    f"kalitini qo'shing (https://console.groq.com/keys).\n\n"
                    f"Texnik xato: `{type(e).__name__}: {str(e)[:400]}`"
                )
        if turn is not None:
            st.session_state.history.append(turn)
            save_history(st.session_state.history)
            increment_usage()


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
        if not rate_limit_gate("contract"):
            st.stop()
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
                try:
                    with st.status("📄 Shartnoma o'qilmoqda va tahlil qilinmoqda…", expanded=ADMIN_MODE) as s:
                        if ADMIN_MODE:
                            st.caption(f"belgi soni: {len(text)}")
                        result = analyze_contract(text)
                        s.update(label="✅ Tahlil tayyor", state="complete", expanded=ADMIN_MODE)
                    st.markdown(result)
                    increment_usage()
                except Exception as e:
                    st.error(
                        f"❌ Tahlil bajarilmadi — LLM provider javob bermadi.\n\n"
                        f"**Iltimos:** biroz kutib qayta urinib ko'ring yoki Groq "
                        f"kalitini qo'shing (https://console.groq.com/keys).\n\n"
                        f"Texnik xato: `{type(e).__name__}: {str(e)[:400]}`"
                    )


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
st.markdown(
    f"💬 **Muammo bo'lsa bog'laning:** "
    f"[{ADMIN_EMAIL}]({_mailto('Legal-Multiagent — muammo/taklif')})"
)
st.caption(
    "⚠️ Bu vosita huquqiy ma'lumot beradi, yuridik maslahat emas. "
    "Muhim qarorlar uchun advokatga murojaat qiling."
)
