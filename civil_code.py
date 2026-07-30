"""Civil Code parsing shared by the Streamlit app and the one-time ingest script.

Kept free of Streamlit imports so ingest.py (and tests) can use it without
starting a script run.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List

import fitz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PARTS = [
    ("Birinchi qism", os.path.join(BASE_DIR, "civil_code.pdf")),
    ("Ikkinchi qism", os.path.join(BASE_DIR, "civil_code_2.pdf")),
]

PROXY_BASE_URL = "https://saidazam-litellm-proxy.hf.space/v1"
COLLECTION_NAME = "civil_code"
EMBED_MODEL = "gemini-embedding"
VECTOR_SIZE = 3072
# Gemini's BatchEmbedContents caps a request at 100 inputs: measured against the
# proxy, 50 and 100 succeed while 200+ return 400 BadRequestError. langchain's
# OpenAIEmbeddings defaults to 1000, which fails outright on a corpus this size
# and then burns minutes in retries, so this ceiling has to be set explicitly.
EMBED_BATCH = 100

SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUP_MAP = str.maketrans("0123456789", SUP_CHARS)
SUP_TO_DIGIT = {c: str(i) for i, c in enumerate(SUP_CHARS)}

ARTICLE_RE = re.compile(r"\b(\d+[" + SUP_CHARS + r"]*)\s*-\s*modda\s*\.")
QUERY_ARTICLE_RE = re.compile(r"(\d+)\s*(?:[-–—]\s*(\d+))?\s*[-–—]?\s*modda", re.IGNORECASE)


def load_pdf_text(path: str) -> str:
    """Extract text, preserving superscript article numbers.

    The code inserts amended articles as 26¹, 173⁷, 358¹ and so on. Plain
    page.get_text() flattens those to "261", "1737", "3581" — which mislabels
    them and, for 26¹, collides head-on with the real article 261 (neustoyka
    shakllari), leaving two different articles under one label. PyMuPDF sets
    bit 0 of span["flags"] on superscript runs, so rebuild span by span.
    """
    doc = fitz.open(path)
    out = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if span["flags"] & 1:
                        text = text.translate(SUP_MAP)
                    out.append(text)
                out.append("\n")
    return "".join(out)


def normalize_article(label: str) -> str:
    """'26¹' -> '26-1', so either spelling a user types resolves to one key."""
    base, sup = [], []
    for ch in label:
        (sup if ch in SUP_TO_DIGIT else base).append(SUP_TO_DIGIT.get(ch, ch))
    return "".join(base) + ("-" + "".join(sup) if sup else "")


def legal_chunk(full_text: str) -> List[dict]:
    hits = list(ARTICLE_RE.finditer(full_text))
    chunks = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(full_text)
        label = m.group(1)
        chunks.append({
            "article": f"{label}-modda",
            "key": normalize_article(label),
            "text": full_text[start:end].strip(),
        })
    return chunks


def load_articles() -> List[dict]:
    """Chunk every Civil Code part that is present, tagging each with its part."""
    articles = []
    for part, path in PDF_PARTS:
        if not os.path.exists(path):
            continue
        for a in legal_chunk(load_pdf_text(path)):
            a["part"] = part
            articles.append(a)
    return articles


def _desuperscript(text: str) -> str:
    """'26¹-modda' -> '26-1-modda' so a pasted superscript matches the same key."""
    out, run = [], []
    for ch in text:
        if ch in SUP_TO_DIGIT:
            run.append(SUP_TO_DIGIT[ch])
            continue
        if run:
            out.append("-" + "".join(run))
            run = []
        out.append(ch)
    if run:
        out.append("-" + "".join(run))
    return "".join(out)


def article_keys_in(question: str) -> List[str]:
    """Article numbers explicitly named in a question, as normalized keys."""
    keys = []
    for m in QUERY_ARTICLE_RE.finditer(_desuperscript(question)):
        base, sup = m.group(1), m.group(2)
        key = f"{base}-{sup}" if sup else base
        if key not in keys:
            keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Embedding the corpus
# ---------------------------------------------------------------------------
EMBED_WORKERS = 5


def _embed_batch(texts: List[str], api_key: str, attempts: int = 3) -> List[List[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    last = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            PROXY_BASE_URL + "/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return [row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:200]
            last = RuntimeError(f"embed HTTP {exc.code}: {body}")
            if exc.code < 500 and exc.code != 429:
                raise last  # a 400 will not fix itself on retry
        except Exception as exc:  # network hiccup
            last = exc
        time.sleep(2 * (attempt + 1))
    raise last


def embed_corpus(texts: List[str], api_key: str,
                 workers: int = EMBED_WORKERS) -> List[List[float]]:
    """Embed every article, running EMBED_BATCH-sized requests concurrently.

    langchain's OpenAIEmbeddings issues its batches strictly one after another,
    which on ~1200 articles measured at roughly six minutes — long enough that a
    cold Streamlit boot is unusable. The proxy handles a handful of concurrent
    requests happily, so fan the batches out and keep the ordering.
    """
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda b: _embed_batch(b, api_key), batches))
    return [vec for batch in results for vec in batch]
