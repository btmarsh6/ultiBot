"""
Ulti Rules Bot — Streamlit chatbot for Ultimate disc sport rules.
Answers rules questions with section citations using Claude Haiku 4.5 + Pinecone RAG.
"""

import re
import csv
import streamlit as st
import anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Ulti Rules Bot", page_icon="🥏", layout="centered")

# --- Constants ---
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "claude-haiku-4-5"
TOP_K = 5
MAX_HISTORY = 10  # sliding window for Claude context

RULESET_META = {
    "USAU": {"edition": "2026-2027", "prefix": "USAU", "namespace": "usau"},
    "WFDF": {"edition": "2025-2028", "prefix": "WFDF", "namespace": "wfdf"},
}

SYSTEM_PROMPT = """\
You are an expert assistant for the rules of Ultimate disc sport. Answer questions \
strictly using the retrieved rule excerpts provided below.

ACTIVE RULESET: {ruleset_name} ({ruleset_edition})

RULE EXCERPTS:
{context_block}

INSTRUCTIONS:
- If the excerpts clearly answer the question, respond concisely (1-4 sentences) \
and cite every relevant rule section in parentheses at the end, e.g. ({ruleset_prefix} 3.I.6).
- If the question is ambiguous and could have meaningfully different answers depending \
on a missing detail, ask one or two focused clarifying questions before answering. Do not answer yet.
- If the excerpts do not contain sufficient information to answer, say so clearly. \
Do not invent rules.
- Keep answers brief — this is used on the sideline during a game.\
"""


# --- Cached resources ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_index():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index(st.secrets["PINECONE_INDEX_NAME"])


@st.cache_resource
def load_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


@st.cache_resource
def load_glossary() -> dict[str, str]:
    glossary = {}
    try:
        with open("data/Slang-Glossary.csv", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                glossary[row["Term"].lower()] = row["Definition"]
    except FileNotFoundError:
        pass
    return glossary


# --- Helpers ---
def expand_slang(query: str, glossary: dict[str, str]) -> str:
    """Append definitions of any slang terms found in the query."""
    for term, definition in glossary.items():
        if re.search(rf"\b{re.escape(term)}\b", query, re.IGNORECASE):
            query += f" ({definition})"
    return query


def get_retrieval_query(messages: list[dict]) -> str:
    """Combine recent user messages for context-aware retrieval.

    When Claude asks a clarifying question and the user replies (e.g. "No"),
    including the original question in the embedding query produces better results.
    """
    recent_user = [m["content"] for m in messages if m["role"] == "user"][-3:]
    return " ".join(recent_user)


def retrieve_context(query: str, namespace: str, index, embedder: SentenceTransformer) -> list:
    vec = embedder.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=vec, top_k=TOP_K, include_metadata=True, namespace=namespace)
    return results.matches


def build_context_block(matches: list) -> str:
    if not matches:
        return "No relevant rule excerpts found."
    parts = [f"[{i + 1}] {m.metadata.get('text', '')}" for i, m in enumerate(matches)]
    return "\n\n".join(parts)


def build_system_prompt(ruleset: str, context_block: str) -> str:
    meta = RULESET_META[ruleset]
    return SYSTEM_PROMPT.format(
        ruleset_name=ruleset,
        ruleset_edition=meta["edition"],
        ruleset_prefix=meta["prefix"],
        context_block=context_block,
    )


def call_claude(messages: list[dict], system_prompt: str, client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        system=system_prompt,
        messages=messages[-MAX_HISTORY:],
    )
    return response.content[0].text


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ruleset" not in st.session_state:
    st.session_state.ruleset = "USAU"

# --- UI ---
st.title("Ulti Rules Bot")
st.caption("Ask a rules question for a quick, cited answer.")

top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    ruleset = st.radio(
        "Ruleset",
        ["USAU", "WFDF"],
        horizontal=True,
        index=0 if st.session_state.ruleset == "USAU" else 1,
        label_visibility="collapsed",
    )
with top_col2:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Clear chat when ruleset is switched
if ruleset != st.session_state.ruleset:
    st.session_state.messages = []
    st.session_state.ruleset = ruleset

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about Ultimate rules..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Load resources
    embedder = load_embedder()
    index = load_index()
    client = load_anthropic_client()
    glossary = load_glossary()

    # Build retrieval query from conversation context, then expand slang
    retrieval_query = get_retrieval_query(st.session_state.messages)
    expanded = expand_slang(retrieval_query, glossary)

    # Retrieve and generate
    namespace = RULESET_META[st.session_state.ruleset]["namespace"]
    matches = retrieve_context(expanded, namespace, index, embedder)
    # print(matches)  # debug
    context_block = build_context_block(matches)
    system_prompt = build_system_prompt(st.session_state.ruleset, context_block)

    with st.chat_message("assistant"):
        with st.spinner(""):
            answer = call_claude(st.session_state.messages, system_prompt, client)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
