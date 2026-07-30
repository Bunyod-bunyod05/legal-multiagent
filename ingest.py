"""One-time ingest: embed the Civil Code into a managed Qdrant collection.

Run this once, and again whenever a PDF or the chunker changes. Afterwards set
QDRANT_URL / QDRANT_API_KEY in the app's secrets and the app stops embedding the
corpus on every cold boot — it only embeds the incoming question.

    set GEMINI_API_KEY=...
    set QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
    set QDRANT_API_KEY=...
    python ingest.py

Pass --recreate to drop an existing collection first, which is what you want
after changing the chunking logic, since stale points would otherwise linger.
"""
import argparse
import os
import sys
from collections import Counter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

# Same chunker the app uses, so the collection can never drift from what the
# app expects — identical superscript handling and identical normalized keys.
from civil_code import (
    COLLECTION_NAME,
    EMBED_BATCH,
    EMBED_MODEL,
    PDF_PARTS,
    PROXY_BASE_URL,
    VECTOR_SIZE,
    legal_chunk,
    load_pdf_text,
)


def load_articles_verbose():
    articles = []
    for part, path in PDF_PARTS:
        if not os.path.exists(path):
            print(f"  ! yo'q, o'tkazib yuborildi: {os.path.basename(path)}")
            continue
        found = legal_chunk(load_pdf_text(path))
        for a in found:
            a["part"] = part
        articles += found
        print(f"  {part:15s} {len(found):5d} modda")
    return articles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="kolleksiyani o'chirib qaytadan yaratish")
    args = ap.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    if not gemini_key or not qdrant_url:
        print("GEMINI_API_KEY va QDRANT_URL environment o'zgaruvchilari kerak.")
        return 1

    print("Chunking...")
    articles = load_articles_verbose()
    if not articles:
        print("Hech qanday modda topilmadi — PDF fayllarni tekshiring.")
        return 1

    keys = [a["key"] for a in articles]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    if dupes:
        print(f"To'xtatildi: takroriy modda kalitlari -> {sorted(dupes)[:10]}")
        return 1
    print(f"  jami {len(articles)} modda, kalitlar unikal")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
    if client.collection_exists(COLLECTION_NAME):
        if not args.recreate:
            count = client.count(COLLECTION_NAME).count
            print(f"'{COLLECTION_NAME}' allaqachon mavjud ({count} nuqta). "
                  "Qayta yuklash uchun --recreate bilan ishga tushiring.")
            return 1
        print(f"'{COLLECTION_NAME}' o'chirilmoqda...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE, distance=models.Distance.COSINE
        ),
    )
    print(f"'{COLLECTION_NAME}' yaratildi ({VECTOR_SIZE} o'lcham, cosine)")

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(
            base_url=PROXY_BASE_URL, api_key=gemini_key,
            model=EMBED_MODEL, chunk_size=EMBED_BATCH,
        ),
    )

    docs = [
        Document(
            page_content=a["text"],
            metadata={"article": a["article"], "key": a["key"], "part": a["part"]},
        )
        for a in articles
    ]
    for i in range(0, len(docs), EMBED_BATCH):
        store.add_documents(docs[i:i + EMBED_BATCH])
        print(f"  {min(i + EMBED_BATCH, len(docs)):5d}/{len(docs)}")

    written = client.count(COLLECTION_NAME).count
    print(f"\nTayyor: {written} nuqta '{COLLECTION_NAME}' kolleksiyasida.")
    if written != len(docs):
        print(f"OGOHLANTIRISH: {len(docs)} kutilgan edi, {written} yozildi.")
        return 1
    print("Endi Streamlit Secrets'ga QDRANT_URL va QDRANT_API_KEY qo'shing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
