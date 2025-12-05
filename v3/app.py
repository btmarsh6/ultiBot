"""
Streamlit RAG Chatbot for Ultimate Rules (WFDF & USAU)

Requirements:
- Run in Python 3.11 virtualenv with your requirements.txt installed.
- .env must contain at least:
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=ultimate-rules
    HF_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
    HF_GENERATION_MODEL=google/flan-t5-small
Optional:
    HF_DEVICE=cuda  # or cpu
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

import streamlit as st

# embeddings
from sentence_transformers import SentenceTransformer

# generation
from transformers import pipeline, logging as hf_logging
hf_logging.set_verbosity_error()  # reduce noisy hf logging

# vector store
from pinecone import Pinecone

load_dotenv()

# ---- Config from .env (with defaults) ----
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ultimate-rules")
EMBED_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
GEN_MODEL = os.environ.get("HF_GENERATION_MODEL", "google/flan-t5-small")
HF_DEVICE = os.environ.get("HF_DEVICE", "cpu")  # "cuda" or "cpu"

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing — add it to your .env file")

# ---- Streamlit page ----
st.set_page_config(page_title="UltiBot", layout="wide")
st.title("UltiBot - Ultimate Rules Q&A Chatbot (WFDF & USAU)")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**Settings**")
    top_k = st.slider("retrieved snippets (top_k)", 1, 8, 4)
    max_tokens = st.slider("generation max tokens", 64, 1024, 256, step=32)
    show_prompt = st.checkbox("Show prompt sent to generator",
                              value=False)
    smart_mode = st.checkbox("Refuse when uncertain (safer answers)",
                             value=True)

with col1:
    query = st.text_input("Ask a question about Ultimate rules (WFDF / USAU):",
                          value="What happens after a contested catch?")
    st.write("")  # spacing


# ---- Caches for model loading ----
@st.cache_resource(show_spinner=False)
def init_pinecone_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    # SentenceTransformers wrapper (BGE-large recommended)
    model = SentenceTransformer(model_name, device=HF_DEVICE)
    return model


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    # pipeline will pick device automatically; prefer "cpu" for small models,
    # "cuda" if GPU available and HF_DEVICE=="cuda"
    gen = pipeline("text2text-generation", model=model_name, device=HF_DEVICE)
    return gen


# Initialize resources
with st.spinner("Initializing models and Pinecone..."):
    pinecone_client = init_pinecone_client(PINECONE_API_KEY)
    index = pinecone_client.Index(INDEX_NAME)
    embedder = load_embedder(EMBED_MODEL)
    generator = load_generator(GEN_MODEL)


# Helper: robust query parsing (handles dict-like or object-like results)
def parse_query_result(res: Any) -> List[Dict[str, Any]]:
    matches = []
    # Pinecone v4 can return a dict or an object with .matches; handle both
    if isinstance(res, dict):
        raw_matches = res.get("matches", [])
    else:
        raw_matches = getattr(res, "matches", None)
        if raw_matches is None:
            # Last fallback: try `res["matches"]` style
            try:
                raw_matches = res["matches"]
            except Exception:
                raw_matches = []
    for m in raw_matches:
        if isinstance(m, dict):
            meta = m.get("metadata") or {}
            matches.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": meta
            })
        else:
            # object-like match (proto)
            metadata = getattr(m, "metadata", None) or {}
            score = getattr(m, "score", None)
            _id = getattr(m, "id", None)
            matches.append({"id": _id, "score": score, "metadata": metadata})
    return matches


# Build the prompt that will be sent to the generator
def build_prompt(query_text: str, contexts: List[Dict[str, Any]], safe_mode: bool = True) -> str:
    ctx_parts = []
    for i, c in enumerate(contexts):
        meta = c.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        text = meta.get("text", "")  # ingest stored snippet or chunk
        # limit snippet length in prompt
        snippet = text if len(text) <= 1200 else text[:1200] + "..."
        ctx_parts.append(f"[{i+1}] Source: {source}\n{snippet}")
    ctx = "\n\n".join(ctx_parts) if ctx_parts else "No context retrieved."
    instruction = (
        "You are an expert on Ultimate (flying disc) rules. Use ONLY the provided excerpts when answering. "
        "If the excerpts do not contain the answer, say you don't know and do not hallucinate."
    )
    if not safe_mode:
        instruction = (
            "You are an expert on Ultimate (flying disc) rules. Use the provided excerpts to answer the question. "
            "If necessary, you may infer likely answers, but mark anything inferred clearly."
        )
    prompt = (
        instruction
        + "\n\nCONTEXT:\n" + ctx
        + f"\n\nQUESTION: {query_text}\n\nAnswer concisely. When quoting or referring to the excerpts, cite the excerpt number(s) in square brackets (e.g., [1], [2])."
    )
    return prompt


# ---- Main interaction ----
if st.button("Ask") and query.strip():
    with st.spinner("Embedding query and retrieving..."):
        # create query embedding; normalize if model returns unnormalized vectors
        q_emb = embedder.encode(query, normalize_embeddings=True).tolist()
        # Query pinecone
        try:
            res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.stop()

        matches = parse_query_result(res)

    if not matches:
        st.warning("No matches found in the index. Did you run the ingestion script and confirm the index name?")
    else:
        st.subheader("Retrieved snippets")
        for i, m in enumerate(matches):
            meta = m.get("metadata", {}) or {}
            source = meta.get("source", "unknown")
            text = meta.get("text", "") or meta.get("chunk", "") or "<no text in metadata>"
            score = m.get("score", None)
            st.markdown(f"**{i+1}. Source:** {source} — _score: {score:.4f}_")
            st.write(text if len(text) < 1000 else text[:1000] + "...")
            st.write("---")

        # Build prompt and run generator
        prompt = build_prompt(query, matches, safe_mode=smart_mode)
        if show_prompt:
            st.subheader("Prompt")
            st.text_area("Prompt sent to model", value=prompt, height=240)

        with st.spinner("Generating answer..."):
            try:
                out = generator(prompt, max_length=max_tokens, do_sample=False)
                # pipeline returns list of dicts
                answer = out[0]["generated_text"] if isinstance(out, list) and out else str(out)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        st.subheader("Answer")
        st.write(answer)

        # Provide quick citation summary
        st.subheader("Citation summary")
        # group sources and excerpt indices
        src_map: Dict[str, List[int]] = {}
        for idx, m in enumerate(matches, start=1):
            meta = m.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            src_map.setdefault(src, []).append(idx)
        for src, idxs in src_map.items():
            st.write(f"- **{src}**: excerpts {idxs}")

# ---- Footer / hints ----
st.write("")
st.info(
    "Hints: If the model hallucinates or misses fine details, try increasing top_k, using a stronger generator model (set HF_GENERATION_MODEL in .env), "
    "or improving chunking in the ingestion step. Large models and BGE-large may be slow on CPU. For best performance use a GPU."
)
