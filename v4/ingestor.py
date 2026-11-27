"""
Ingestion script. Run locally before deploying the app.
Embeds USAU and WFDF rule PDFs into Pinecone using separate namespaces.

Pipeline:
    1. Extract text per page, strip page numbers, collapse broken whitespace
       (pypdf emits a newline after almost every word in the USAU PDF).
    2. Split the full document into rule units at rule IDs
       (USAU: 19.A.1.a. / appendix C1.A.2.  WFDF: 18.3.1.1.).
    3. Pack consecutive units of the same top-level section into chunks,
       preferring to break at second-level boundaries (19.A -> 19.B).
    4. Store section IDs, section title, and page in metadata.

Setup:
    pip install pypdf langchain-text-splitters sentence-transformers pinecone python-dotenv

Create a .env file in this directory:
    PINECONE_API_KEY=your-key-here
    PINECONE_INDEX_NAME=ultimate-rules-v4

Usage:
    python ingestor.py --dry-run     # build chunks, write chunk_preview_<ns>.txt, no upload
    python ingestor.py               # REPLACES all vectors in the usau/wfdf namespaces
"""

import argparse
import bisect
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ultimate-rules-v4")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DIMENSION = 768
BATCH_SIZE = 100

# Chunk packing (characters). bge-base has a 512-token limit (~2000 chars).
MIN_CHUNK = 400    # below this, keep adding the next second-level section
MAX_CHUNK = 1500   # never exceed this

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# Rule ID patterns. Each must be followed by whitespace in the cleaned text.
USAU_ID = re.compile(
    r"(?<!\S)((?:[A-G]\d{1,2}|\d{1,2})\.(?:[A-Z]\.(?:\d{1,2}\.(?:[a-z]\.(?:\d{1,2}\.)?)?)?)?)(?=\s)"
)
WFDF_ID = re.compile(r"(?<!\S)(\d{1,2}\.(?:\d{1,2}\.){0,3})(?=\s)")

RULESETS = {
    "usau": {
        "label": "USAU",
        "pdf": DATA_DIR / "USAU-Rules-of-Ultimate-2026-2027.pdf",
        "id_pattern": USAU_ID,
        "start_marker": "Preface Ultimate is",
        "end_marker": None,
        # Unnumbered headings: (regex, section id, title)
        "headings": [
            (re.compile(r"Preface (?=Ultimate is)"), "Preface", "Preface"),
            (re.compile(r"Appendix A: Field Diagram"), "Appendix A", "Field Diagram"),
            (re.compile(r"Appendix B: Mixed Rules and Adaptations"), "Appendix B", "Mixed Rules and Adaptations"),
            (re.compile(r"Appendix C: Misconduct System"), "Appendix C", "Misconduct System"),
            (re.compile(r"Appendix D: Hand Signals"), "Appendix D", "Hand Signals"),
            (re.compile(r"Appendix E: Youth Rules Adaptations"), "Appendix E", "Youth Rules Adaptations"),
            (re.compile(r"Appendix F: Beach Ultimate Rules Adaptations"), "Appendix F", "Beach Ultimate Rules Adaptations"),
            (re.compile(r"Appendix G: Ultimate 4.s Rules Adaptations"), "Appendix G", "Ultimate 4's Rules Adaptations"),
        ],
    },
    "wfdf": {
        "label": "WFDF",
        "pdf": DATA_DIR / "WFDF-Rules-of-Ultimate-2025-2028.pdf",
        "id_pattern": WFDF_ID,
        "start_marker": "Introduction Ultimate is",
        "end_marker": "Legal License This Work",
        "headings": [
            (re.compile(r"Introduction (?=Ultimate is)"), "Introduction", "Introduction"),
            (re.compile(r"Definitions (?=Act of throwing)"), "Definitions", "Definitions"),
        ],
    },
}


@dataclass
class Unit:
    """One rule (ID + its text) or an unnumbered heading block."""
    rule_id: str          # e.g. "19.A.1", "C1.A", "18.3.1", "Definitions"
    top: str              # top-level section key, e.g. "19", "Appendix C", "Definitions"
    group: str            # second-level key used for chunk boundaries, e.g. "19.A"
    text: str
    page: int


@dataclass
class Chunk:
    top: str
    title: str
    units: list[Unit] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(u.text for u in self.units)


# --- 1. Extraction & cleanup ---
def clean_page(text: str, page_num: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Strip a leading or trailing page number equal to this page's number
    text = re.sub(rf"^{page_num}\s+", "", text)
    text = re.sub(rf"\s+{page_num}$", "", text)
    return text


def extract_document(pdf_path: Path, start_marker: str, end_marker: str | None):
    """Return (full_text, page_starts) where page_starts[i] is the char offset of page i+1."""
    reader = PdfReader(pdf_path)
    parts, page_starts, offset = [], [], 0
    for i, page in enumerate(reader.pages):
        cleaned = clean_page(page.extract_text() or "", i + 1)
        page_starts.append(offset)
        parts.append(cleaned)
        offset += len(cleaned) + 1
    full = " ".join(parts)

    start = full.find(start_marker)
    if start < 0:
        raise ValueError(f"Start marker {start_marker!r} not found in {pdf_path.name}")
    end = full.find(end_marker, start) if end_marker else -1
    end = end if end >= 0 else len(full)
    return full, page_starts, start, end


def page_at(offset: int, page_starts: list[int]) -> int:
    return bisect.bisect_right(page_starts, offset)


# --- 2. Split into rule units ---
def id_parts(rule_id: str) -> list[str]:
    return [p for p in rule_id.rstrip(".").split(".") if p]


def find_boundaries(text: str, start: int, end: int, cfg: dict) -> list[tuple[int, int, str, str]]:
    """Find (pos, text_start, rule_id, top) for every valid rule ID / heading in [start, end).

    Top-level numbers must appear in sequence (1, 2, 3...) and sub-rules must belong to
    the current top-level section; this filters out cross-references like "see 17.C."
    """
    candidates = []
    for pattern, rule_id, _title in cfg["headings"]:
        for m in pattern.finditer(text, start, end):
            candidates.append((m.start(), m.start(), rule_id, rule_id))
    for m in cfg["id_pattern"].finditer(text, start, end):
        rid = m.group(1).rstrip(".")
        candidates.append((m.start(), m.end(), rid, None))
    candidates.sort()

    boundaries = []
    current_top = None   # e.g. "19", "Appendix C", "Definitions"
    last_num = 0
    numeric = [c for c in candidates if c[3] is None]
    next_numeric_head = {}
    for a, b in zip(numeric, numeric[1:]):
        next_numeric_head[a[0]] = id_parts(b[2])[0]

    for pos, id_end, rid, heading_top in candidates:
        tstart = pos
        if heading_top:  # unnumbered heading
            current_top = heading_top
            boundaries.append((pos, tstart, rid, heading_top))
            continue
        parts = id_parts(rid)
        head = parts[0]
        if head.isdigit():
            num = int(head)
            if len(parts) == 1:
                # New top-level section: must be next in sequence, followed by a title,
                # and the next rule ID must belong to it (e.g. "8." followed by "8.A.")
                following = text[id_end : id_end + 3].lstrip()
                if (
                    num == last_num + 1
                    and following[:1].isupper()
                    and next_numeric_head.get(pos) == head
                ):
                    current_top, last_num = head, num
                    boundaries.append((pos, tstart, rid, head))
            elif head == current_top:
                boundaries.append((pos, tstart, rid, head))
        else:
            # Appendix rule like C1.A -> belongs to "Appendix C"
            appendix = f"Appendix {head[0]}"
            if current_top == appendix:
                boundaries.append((pos, tstart, rid, appendix))
    return boundaries


def split_units(full: str, page_starts: list[int], start: int, end: int, cfg: dict) -> tuple[list[Unit], dict]:
    bounds = find_boundaries(full, start, end, cfg)
    titles = {rid: title for _p, rid, title in cfg["headings"]}
    heading_ids = set(titles)
    units = []
    for i, (pos, tstart, rid, top) in enumerate(bounds):
        stop = bounds[i + 1][0] if i + 1 < len(bounds) else end
        text = full[tstart:stop].strip()
        parts = id_parts(rid)
        if rid in heading_ids or len(parts) == 1:
            group = rid
        else:
            group = ".".join(parts[:2])
        # Top-level numbered section: capture its title ("19. Picks" -> "Picks")
        if parts[0].isdigit() and len(parts) == 1:
            titles[top] = text[len(rid) + 1 :].strip()[:60]
        units.append(Unit(rid, top, group, text, page_at(tstart, page_starts)))
    return units, titles


# --- 3. Pack units into chunks ---
_long_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK - 200,
    chunk_overlap=150,
    separators=["]] ", ". ", "; ", " ", ""],
    keep_separator="end",
)


def split_long_unit(u: Unit) -> list[Unit]:
    if len(u.text) <= MAX_CHUNK - 200:
        return [u]
    return [Unit(u.rule_id, u.top, u.group, piece, u.page) for piece in _long_splitter.split_text(u.text)]


def pack_chunks(units: list[Unit], titles: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: Chunk | None = None
    for unit in units:
        for u in split_long_unit(unit):
            size = len(cur.body) if cur else 0
            new_chunk = (
                cur is None
                or u.top != cur.top                                            # never cross top-level sections
                or size + len(u.text) + 1 > MAX_CHUNK                          # too big
                or (u.group != cur.units[-1].group and size >= MIN_CHUNK)      # clean break at 19.A -> 19.B
            )
            if new_chunk:
                cur = Chunk(top=u.top, title=titles.get(u.top, ""))
                chunks.append(cur)
            cur.units.append(u)

    # Fold small leftover chunks into the previous chunk of the same section if they fit
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if (
            prev
            and len(c.body) < MIN_CHUNK
            and prev.top == c.top
            and len(prev.body) + len(c.body) + 1 <= MAX_CHUNK
        ):
            prev.units.extend(c.units)
        else:
            merged.append(c)
    return merged


def chunk_header(label: str, chunk: Chunk) -> str:
    top = chunk.top if not chunk.top.isdigit() else f"Section {chunk.top}"
    title = f" – {chunk.title}" if chunk.title and chunk.title != chunk.top else ""
    return f"[{label} {top}{title}]"


def build_records(ns: str, cfg: dict) -> list[dict]:
    full, page_starts, start, end = extract_document(cfg["pdf"], cfg["start_marker"], cfg["end_marker"])
    units, titles = split_units(full, page_starts, start, end, cfg)
    chunks = pack_chunks(units, titles)

    records = []
    for i, c in enumerate(chunks):
        ids = list(dict.fromkeys(u.rule_id for u in c.units))  # unique, ordered
        text = f"{chunk_header(cfg['label'], c)}\n{c.body}"
        records.append({
            "id": f"{ns}-{i:04d}",
            "text": text,
            "metadata": {
                "text": text,
                "ruleset": cfg["label"],
                "section": ids[0],
                "sections": ids,
                "section_title": c.title,
                "top_section": c.top,
                "page": c.units[0].page,
                "page_end": c.units[-1].page,
            },
        })
    return records


# --- 4. Embed & upload ---
def write_preview(ns: str, records: list[dict]) -> Path:
    path = HERE / f"chunk_preview_{ns}.txt"
    sizes = [len(r["text"]) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(records)} chunks | min {min(sizes)} | avg {sum(sizes) // len(sizes)} | max {max(sizes)} chars\n\n")
        for r in records:
            m = r["metadata"]
            f.write(f"===== {r['id']} | sections {', '.join(m['sections'])} | pages {m['page']}-{m['page_end']} | {len(r['text'])} chars\n")
            f.write(r["text"] + "\n\n")
    return path


def upsert(records: list[dict], index, namespace: str, model) -> None:
    embeddings = model.encode([r["text"] for r in records], normalize_embeddings=True, show_progress_bar=True)
    vectors = [
        {"id": r["id"], "values": e.tolist(), "metadata": r["metadata"]}
        for r, e in zip(records, embeddings)
    ]
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + BATCH_SIZE], namespace=namespace)
        print(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)} vectors")


def get_index():
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")
    pc = Pinecone(api_key=api_key)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
    else:
        print(f"Using existing Pinecone index '{INDEX_NAME}'")
    return pc.Index(INDEX_NAME)


def main():
    parser = argparse.ArgumentParser(description="Chunk rule PDFs by section and load into Pinecone.")
    parser.add_argument("--dry-run", action="store_true", help="Build chunks and write previews only; no upload")
    args = parser.parse_args()

    all_records = {}
    for ns, cfg in RULESETS.items():
        print(f"\nChunking {cfg['label']} from {cfg['pdf'].name}...")
        records = build_records(ns, cfg)
        # preview = write_preview(ns, records)      # preview files for inspection/debugging
        # print(f"  {len(records)} chunks -> preview written to {preview.name}")
        all_records[ns] = records

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return

    from sentence_transformers import SentenceTransformer

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    index = get_index()
    existing = index.describe_index_stats().namespaces

    for ns, records in all_records.items():
        if ns in existing:
            print(f"\nDeleting {existing[ns].vector_count} existing vectors in '{ns}'...")
            index.delete(delete_all=True, namespace=ns)
        print(f"Uploading {len(records)} chunks to '{ns}'...")
        upsert(records, index, ns, model)

    time.sleep(5)  # stats are eventually consistent
    print("\nIngestion complete!")
    for ns, info in index.describe_index_stats().namespaces.items():
        print(f"  {ns}: {info.vector_count} vectors")


if __name__ == "__main__":
    main()
