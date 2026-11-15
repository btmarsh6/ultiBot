"""
One-time ingestion script. Run locally before deploying the app.
Embeds USAU and WFDF rule PDFs into Pinecone using separate namespaces.

Setup:
    pip install pypdf langchain-text-splitters sentence-transformers pinecone python-dotenv

Create a .env file in this directory:
    PINECONE_API_KEY=your-key-here
    PINECONE_INDEX_NAME=ultimate-rules-v4

Usage:
    python ingestor.py

NOTE: Run only once. Running again will add duplicate vectors.
"""

import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ultimate-rules-v4")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DIMENSION = 768
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
BATCH_SIZE = 100

DATA_DIR = Path(__file__).parent / "data"
RULESETS = {
    "usau": DATA_DIR / "USAU-Rules-of-Ultimate-2026-2027.pdf",
    "wfdf": DATA_DIR / "WFDF-Rules-of-Ultimate-2025-2028.pdf",
}


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if split.strip():
                chunks.append({"text": split.strip(), "page": page["page"]})
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def upsert_to_pinecone(chunks: list[dict], index, namespace: str) -> None:
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": chunk["embedding"],
            "metadata": {"text": chunk["text"], "page": chunk["page"]},
        }
        for chunk in chunks
    ]
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        end = min(i + BATCH_SIZE, len(vectors))
        print(f"  Upserted {end}/{len(vectors)} vectors")


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    pc = Pinecone(api_key=api_key)

    existing_names = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_names:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
    else:
        print(f"Using existing Pinecone index '{INDEX_NAME}'")

    index = pc.Index(INDEX_NAME)

    for namespace, pdf_path in RULESETS.items():
        print(f"\nIngesting {namespace.upper()} rules from {pdf_path.name}...")
        if not pdf_path.exists():
            print(f"  ERROR: {pdf_path} not found, skipping.")
            continue

        pages = extract_pages(pdf_path)
        print(f"  Extracted {len(pages)} pages")

        chunks = chunk_pages(pages)
        print(f"  Created {len(chunks)} chunks")

        chunks = embed_chunks(chunks, model)
        upsert_to_pinecone(chunks, index, namespace)
        print(f"  Done ingesting {namespace.upper()}")

    print("\nIngestion complete!")
    stats = index.describe_index_stats()
    for ns, info in stats.namespaces.items():
        print(f"  {ns}: {info.vector_count} vectors")


if __name__ == "__main__":
    main()
