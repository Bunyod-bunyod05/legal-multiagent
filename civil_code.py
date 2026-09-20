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

COLLECTION_NAME = "civil_code"
# Google's text-embedding-004 model (used via langchain-google-genai).
EMBED_MODEL = "models/text-embedding-004"
VECTOR_SIZE = 768
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
    """Chunk every Civil Code part that is present, tagging each with its part.

    Raises rather than returning a short list: silently skipping a missing or
    unparsable PDF produced an app that answered "bu haqda ma'lumot yo'q" to
    every question, with nothing anywhere saying the corpus was empty.
    """
    articles = []
    notes = []
    for part, path in PDF_PARTS:
        name = os.path.basename(path)
        if not os.path.exists(path):
            notes.append(f"{name}: fayl topilmadi")
            continue
        found = legal_chunk(load_pdf_text(path))
        if not found:
            notes.append(f"{name}: matn o'qildi, lekin modda topilmadi")
            continue
        for a in found:
            a["part"] = part
            articles.append(a)

    if not articles:
        listing = ", ".join(sorted(os.listdir(BASE_DIR))) if os.path.isdir(BASE_DIR) else "(yo'q)"
        raise RuntimeError(
            "Fuqarolik kodeksi yuklanmadi. "
            + "; ".join(notes)
            + f" | BASE_DIR={BASE_DIR} | mavjud fayllar: {listing}"
        )
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


def _embed_batch_google(texts: List[str], api_key: str, attempts: int = 3) -> List[List[float]]:
    """Embed a batch via the Google Generative AI REST endpoint directly."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:batchEmbedContents"
        f"?key={api_key}"
    )
    requests_body = [
        {"model": EMBED_MODEL, "content": {"parts": [{"text": t}]}}
        for t in texts
    ]
    payload = json.dumps({"requests": requests_body}).encode()
    last = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            return [e["values"] for e in data["embeddings"]]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:200]
            last = RuntimeError(f"embed HTTP {exc.code}: {body}")
            if exc.code < 500 and exc.code != 429:
                raise last
        except Exception as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise last


def embed_corpus(texts: List[str], api_key: str,
                 workers: int = EMBED_WORKERS) -> List[List[float]]:
    """Embed every article, running EMBED_BATCH-sized requests concurrently.

    Uses the Google Generative AI batchEmbedContents endpoint directly.
    """
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda b: _embed_batch_google(b, api_key), batches))
    return [vec for batch in results for vec in batch]
