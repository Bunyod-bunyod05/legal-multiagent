"""Civil Code parsing shared by the Streamlit app and the one-time ingest script.

Kept free of Streamlit imports so ingest.py (and tests) can use it without
starting a script run.
"""
import os
import re
from typing import List

import fitz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PARTS = [
    ("Birinchi qism", os.path.join(BASE_DIR, "civil_code.pdf")),
    ("Ikkinchi qism", os.path.join(BASE_DIR, "civil_code_2.pdf")),
]

COLLECTION_NAME = "civil_code"
# Local, free, multilingual embeddings via fastembed. Handles Uzbek/Russian
# well and runs entirely on CPU, so no API key or quota is needed.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
EMBED_BATCH = 64

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
# Embedding the corpus (fastembed, local ONNX inference — no API key needed)
# ---------------------------------------------------------------------------


def embed_corpus(texts: List[str]) -> List[List[float]]:
    """Embed every article locally via fastembed. Multilingual, free, offline."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL)
    return [list(v) for v in model.embed(texts, batch_size=EMBED_BATCH)]
