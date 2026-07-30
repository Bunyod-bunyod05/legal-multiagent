"""
Multi-Agent Legal Analyst — Streamlit app (Streamlit Community Cloud).

Switched from Gradio/HF Spaces to Streamlit Community Cloud because
Hugging Face changed its policy: Gradio/Docker Spaces now require a
paid (PRO) plan; only Static Spaces remain free. Streamlit Community
Cloud is still genuinely free for public apps as of 2026.

Same graph logic as the Colab notebook (Supervisor + Retriever + Web +
Code agents). All Gemini calls go through the LiteLLM
proxy, never directly to Google.

Expects `civil_code.pdf` to be present in the same directory (part of
the GitHub repo) — ingested lazily on the first question (cached).
"""
import os
import io
import contextlib
import threading
from typing import TypedDict, List, Optional

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient, models

from civil_code import (
    COLLECTION_NAME,
    EMBED_BATCH,
    EMBED_MODEL,
    PROXY_BASE_URL,
    VECTOR_SIZE,
    article_keys_in,
    embed_corpus,
    load_articles,
)

# ---------------------------------------------------------------------------
# set_page_config MUST be the very first Streamlit call in the script — any
# earlier st.* element (even a cache spinner) makes it raise
# StreamlitAPIException, which crashed the app on every cold boot.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Legal Analyst", page_icon="⚖️")

# ---------------------------------------------------------------------------
# Config — reads from Streamlit Cloud "Secrets" (st.secrets) when deployed,
# or from a local .env-loaded environment variable when run locally.
# Never hardcoded, never printed/displayed.
# ---------------------------------------------------------------------------
def _get_secret(name: str) -> str:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return ""


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# When a managed Qdrant is configured the corpus is embedded once by ingest.py
# and the app only queries it. Without these the app falls back to embedding
# every article in memory on each cold boot, which is ~1200 articles per wake.
QDRANT_URL = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY topilmadi. Streamlit Cloud'da Settings → Secrets bo'limiga qo'shing.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llms_and_embeddings():
    # The proxy key is scoped to ['flash-lite', 'gemini-flash-lite', 'gemini-embedding'].
    # Asking for "gemini-flash" comes back as 403 key_model_access_denied, which
    # killed the supervisor node — the graph's entry point — on every question.
    llm_flash = ChatOpenAI(
        base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model="gemini-flash-lite", temperature=0,
    )
    llm_lite = ChatOpenAI(
        base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model="gemini-flash-lite", temperature=0,
    )
    # chunk_size must stay at or below Gemini's 100-input batch limit — see
    # EMBED_BATCH. The library default of 1000 fails with a 400 and then retries.
    embeddings = OpenAIEmbeddings(
        base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model=EMBED_MODEL,
        chunk_size=EMBED_BATCH,
    )
    return llm_flash, llm_lite, embeddings


# ---------------------------------------------------------------------------
# Retrieval — cached so the corpus is prepared once per app session, and pulled
# lazily from the agent nodes so a cold boot paints the UI straight away.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Fuqarolik kodeksi tayyorlanmoqda (1- va 2-qism)...")
def get_retriever():
    # CORPUS_BUILD is part of this function's body on purpose: st.cache_resource
    # keys on the decorated function's own code, so fixes inside civil_code.py
    # do not invalidate a previously cached (and possibly broken) corpus. Bump it
    # whenever the corpus or its parsing changes.
    CORPUS_BUILD = 2
    _ = CORPUS_BUILD
    _, _, embeddings = get_llms_and_embeddings()

    # Chunking is local (no API calls) and the exact-article lookup needs the
    # full texts anyway, so it runs in both modes.
    articles = load_articles()
    by_key = {a["key"]: a for a in articles}

    # ~1200 articles across both parts, so k=4 left too little context for the
    # generator to reason across neighbouring articles.
    search_kwargs = {"k": 6}

    if QDRANT_URL:
        # Corpus was embedded once by ingest.py; only the query is embedded here.
        store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
        return store.as_retriever(search_kwargs=search_kwargs), by_key, len(articles)

    # No managed Qdrant: embed the corpus into an in-memory collection. Doing it
    # through QdrantVectorStore.from_documents() would serialise the batches and
    # take ~6 minutes on both parts, so embed concurrently and upsert the vectors
    # directly. Payload keys match QdrantVectorStore's defaults so queries work.
    vectors = embed_corpus([a["text"] for a in articles], GEMINI_API_KEY)
    client = QdrantClient(location=":memory:")
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE, distance=models.Distance.COSINE
        ),
    )
    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=vector,
                payload={
                    "page_content": a["text"],
                    "metadata": {
                        "article": a["article"], "key": a["key"], "part": a["part"],
                    },
                },
            )
            for i, (a, vector) in enumerate(zip(articles, vectors))
        ],
    )
    store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    return store.as_retriever(search_kwargs=search_kwargs), by_key, len(articles)


# ---------------------------------------------------------------------------
# Agent state + nodes (identical logic to the Colab notebook)
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
Hozirgacha bajarilgan qadamlar: {state['steps']}
Hozirgacha yig'ilgan ma'lumot: hujjatlar={bool(state['documents'])}, kod natijasi={state['code_result']}

Qaysi agent keyingi navbatda ishlashi kerak?
- 'retriever': agar savol Fuqarolik kodeksi moddasi haqida bo'lsa
- 'code': agar savol hisob-kitob (penya, jarima, foiz, summalar) talab qilsa
- 'web': agar savol kodeksda yo'q, internetdan qidirish kerak bo'lsa
- 'finish': agar javob uchun yetarli ma'lumot yig'ilgan bo'lsa

Bir marta 'code'/'retriever'/'web' ishlatilgan bo'lsa, qayta ishlatma — 'finish' deb qaytar."""
    result = llm_flash.with_structured_output(Route).invoke(prompt)
    return {"plan": result.next, "steps": state["steps"] + [f"supervisor→{result.next}"]}


def route_after_supervisor(state):
    return state["plan"]


def run_sandboxed_code(code_str: str, timeout_sec: int = 5) -> str:
    """Run the generated arithmetic in a restricted namespace, with a time cap.

    Uses a worker thread rather than signal.alarm: Streamlit runs the script
    in a ScriptRunner thread, and signal.signal() off the main thread raises
    ValueError (SIGALRM also does not exist on Windows for local runs).
    """
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
    prompt = f"""Quyidagi savol uchun Python kodi yoz — hisob-kitobni bajarib,
print() orqali natijani chiqar. Faqat oddiy arifmetika ishlat (import kerak emas).

Savol: {state['question']}

Agar savolda summalar/foizlar/kunlar aniq ko'rsatilmagan bo'lsa, O'zbekiston
Fuqarolik kodeksi 326-330-moddalaridagi odatiy penya/neustoyka mantig'idan
(kunlik foiz × kechikkan kunlar × summa) foydalanib, taxminiy hisoblash yoz.

Faqat Python kodini qaytar, boshqa izoh yozma."""
    raw_code = llm_lite.invoke(prompt).content
    raw_code = raw_code.replace("```python", "").replace("```", "").strip()
    result = run_sandboxed_code(raw_code)
    return {"code_result": result, "steps": state["steps"] + ["code"]}


def retriever_agent(state):
    retriever, by_key, n_articles = get_retriever()
    texts, cited = [], []

    # An explicit "239-modda" is a lookup, not a similarity problem: cosine
    # search over hundreds of near-identical legal paragraphs frequently ranks
    # the requested article below its neighbours. Resolve it exactly first.
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

    # Carry the corpus size into the visible trace: an empty or half-loaded
    # corpus otherwise looks identical to a question the code simply does not
    # cover, which is exactly how the missing-PDF bug stayed invisible.
    return {
        "documents": state["documents"] + texts,
        "citations": state["citations"] + cited,
        "steps": state["steps"] + [f"retriever({n_articles} modda, {len(cited)} topildi)"],
    }


def web_agent(state):
    # The result-count kwarg is max_results, not k — and because the failure is
    # swallowed below, passing the wrong name made web search look like it had
    # simply found nothing. Surface the error in `steps` instead of hiding it.
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
    prompt = f"""Savol: {state['question']}

Kontekst (topilgan moddalar):
{context}
{code_part}

Faqat yuqoridagi ma'lumotlarga tayanib, aniq va qisqa javob yoz.
Agar hisoblash natijasi bo'lsa, uni javobga aniq kiritib ko'rsat."""
    answer = llm_lite.invoke(prompt).content
    return {"answer": answer, "steps": state["steps"] + ["generate"]}


@st.cache_resource(show_spinner=False)
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
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("⚖️ Multi-Agent Legal Analyst")
st.caption(
    "Supervisor + Retriever + Web + Code agentlar · Fuqarolik kodeksining "
    "1- va 2-qismi (~1200 modda) ustida ishlaydi. Barcha LLM chaqiruvlari proksi orqali."
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_turn(turn):
    st.write(turn["answer"])
    if turn.get("citations"):
        st.caption("Manbalar: " + " · ".join(turn["citations"]))
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
        with st.spinner("Qidirilmoqda..."):
            result = build_graph().invoke(
                {
                    "question": question, "plan": "", "documents": [],
                    "code_result": None, "answer": "", "steps": [], "citations": [],
                },
                {"recursion_limit": 15},
            )
        turn = {
            "question": question, "answer": result["answer"],
            "steps": result["steps"], "citations": result.get("citations", []),
        }
        render_turn(turn)

    st.session_state.history.append(turn)

st.caption(
    "⚠️ Bu vosita huquqiy ma'lumot beradi, yuridik maslahat emas. "
    "Muhim qarorlar uchun advokatga murojaat qiling."
)
