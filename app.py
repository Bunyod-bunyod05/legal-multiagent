"""
Multi-Agent Legal Analyst — Gradio app for Hugging Face Spaces.

Same architecture as the Colab notebook (Supervisor + Retriever + Web +
Code agents), wrapped in a Gradio chat interface. All Gemini calls go
through the instructor's LiteLLM proxy (OpenAI-compatible), never
directly to Google.

Expects `civil_code.pdf` to be present in the same directory (bundled
in the Space repo) — ingested once at startup, not per request.
"""
import os
import re
import io
import contextlib
import signal
from typing import TypedDict, List, Optional

import gradio as gr
import fitz
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Config — keys come from HF Spaces "Secrets" (Settings tab), never hardcoded.
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
PROXY_BASE_URL = "https://saidazam-litellm-proxy.hf.space/v1"
PDF_PATH = os.path.join(os.path.dirname(__file__), "civil_code.pdf")

os.environ.setdefault("TAVILY_API_KEY", TAVILY_API_KEY)

llm_flash = ChatOpenAI(
    base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model="gemini-flash", temperature=0,
)
llm_lite = ChatOpenAI(
    base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model="gemini-flash-lite", temperature=0,
)
embeddings = OpenAIEmbeddings(
    base_url=PROXY_BASE_URL, api_key=GEMINI_API_KEY, model="gemini-embedding",
)

# ---------------------------------------------------------------------------
# Ingest (runs once at app startup)
# ---------------------------------------------------------------------------
def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text


def legal_chunk(full_text: str) -> List[dict]:
    pattern = re.compile(r"\b(\d+)\s*-\s*modda\s*\.")
    hits = list(pattern.finditer(full_text))
    chunks = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(full_text)
        chunks.append({"article": f"{m.group(1)}-modda", "text": full_text[start:end].strip()})
    return chunks


print("Ingest boshlandi...")
full_text = load_pdf_text(PDF_PATH)
articles = legal_chunk(full_text)
docs = [Document(page_content=a["text"], metadata={"article": a["article"]}) for a in articles]
vectorstore = QdrantVectorStore.from_documents(
    docs, embeddings, location=":memory:", collection_name="civil_code"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(f"Ingest tugadi: {len(docs)} ta modda yuklandi.")


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


class Route(BaseModel):
    next: str = Field(description="'retriever', 'web', 'code', yoki 'finish'")


def supervisor(state):
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
    class TimeoutErr(Exception):
        pass

    def _handler(signum, frame):
        raise TimeoutErr("Vaqt limitidan oshdi")

    allowed_builtins = {
        "print": print, "range": range, "len": len, "round": round,
        "abs": abs, "min": min, "max": max, "sum": sum,
        "int": int, "float": float, "str": str,
    }
    safe_globals = {"__builtins__": allowed_builtins}
    buf = io.StringIO()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code_str, safe_globals, {})
    except TimeoutErr as e:
        return f"XATOLIK: {e}"
    except Exception as e:
        return f"XATOLIK: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
    return buf.getvalue().strip()


def code_agent(state):
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
    hits = retriever.invoke(state["question"])
    texts = [f"[{d.metadata.get('article')}] {d.page_content}" for d in hits]
    return {"documents": state["documents"] + texts, "steps": state["steps"] + ["retriever"]}


def web_agent(state):
    try:
        tavily = TavilySearchResults(k=3)
        hits = tavily.invoke({"query": state["question"]})
        texts = [h["content"] for h in hits]
    except Exception:
        texts = []
    return {"documents": state["documents"] + texts, "steps": state["steps"] + ["web"]}


def generate(state):
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
app_graph = g.compile()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def chat_fn(message, history):
    result = app_graph.invoke(
        {
            "question": message, "plan": "", "documents": [],
            "code_result": None, "answer": "", "steps": [],
        },
        {"recursion_limit": 15},
    )
    steps_label = " → ".join(result["steps"])
    return f"{result['answer']}\n\n_Bosqichlar: {steps_label}_"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Multi-Agent Legal Analyst — Fuqarolik kodeksi",
    description=(
        "Supervisor + Retriever + Web + Code agentlar. "
        "Savolingizni yozing — kerak bo'lsa moddadan, internetdan qidiriladi "
        "yoki penya/jarima hisoblanadi."
    ),
    examples=[
        "239-modda nima haqida?",
        "5,000,000 so'm qarz bo'yicha 30 kun kechiktirilgan, kunlik penya 0.1%. Necha so'm penya to'planadi?",
    ],
)

if __name__ == "__main__":
    demo.launch()
