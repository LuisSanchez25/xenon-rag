"""Web interface for the XENON software assistant.

    streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1

Bind to 127.0.0.1, not 0.0.0.0 -- this should not be reachable from the wider
network. Reach it from your laptop through an SSH tunnel:

    ssh -L 8501:localhost:8501 you@fried.rice.edu

Two design decisions worth stating, because both are about trust rather than
features.

The retrieved excerpts are shown, with their scores and links to the exact
lines on GitHub. A scientist who can see what the system read can judge whether
to believe the answer; one who cannot has to take it on faith. Since the
measured answer accuracy is around 86%, taking it on faith is not reasonable,
and hiding the sources would be the wrong call even if it looked tidier.

Questions are logged locally. Not for analytics -- the questions people
actually type are the best source of evaluation cases, and every real question
here is one that does not have to be invented later.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xenonrag.embed import get_embedder            # noqa: E402
from xenonrag.index import VectorIndex             # noqa: E402
from xenonrag.llm import LLMError, get_backend     # noqa: E402
from xenonrag.prompt import build_prompt, estimate_tokens, permalink  # noqa: E402
from xenonrag.retrieve import build_retriever      # noqa: E402


INDEX_DIR = Path("build")
EMBED_MODEL = "bge-small"
QUERY_LOG = Path("logs/queries.jsonl")

EXAMPLES = [
    "How do I load data for a run?",
    "What does s2_min_pmts control?",
    "How does peaklet classification work?",
    "What do I implement to write my own plugin?",
    "What does 'Cannot write to zipfiles' mean?",
]

CITATION = re.compile(r"\[([\w./-]+\.\w+:\d+(?:-\d+)?)\]")


# --------------------------------------------------------------------------
# loading (cached so the model and index load once per session, not per query)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_index(index_dir: str, embed_model: str):
    embedder = get_embedder(embed_model)
    index = VectorIndex.load(Path(index_dir), embedder)
    return index, embedder


@st.cache_resource(show_spinner="Preparing retriever...")
def load_retriever(_index, _embedder, name: str, index_dir: str):
    # index_dir is in the signature only so the cache key changes when the
    # index does; the underscored args are not hashed.
    return build_retriever(name, _index, _embedder)


def link_citations(text: str, sources: list[dict]) -> str:
    """Turn [repo/path:lines] in the answer into clickable links."""
    by_path = {}
    for c in sources:
        url = permalink(c)
        if url:
            by_path.setdefault(f"{c['repo']}/{c['path']}", url)

    def sub(m):
        cite = m.group(1)
        path = cite.rsplit(":", 1)[0]
        url = by_path.get(path)
        return f"[[{cite}]]({url})" if url else m.group(0)

    return CITATION.sub(sub, text)


def log_query(question: str, retriever: str, model: str, n_sources: int):
    """Append the question to a local log. No answers, no identifiers."""
    try:
        QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with QUERY_LOG.open("a") as f:
            f.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": question, "retriever": retriever,
                "model": model, "n_sources": n_sources,
            }) + "\n")
    except OSError:
        pass          # logging must never break a query


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

st.set_page_config(page_title="XENON software assistant",
                   page_icon="[?]", layout="wide")

st.title("XENON software assistant")
st.caption("Answers questions about strax, straxen, xedocs and rframe from the "
           "source and documentation. It quotes what it read -- check the "
           "excerpts before acting on an answer.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Model", ["ollama", "gemini"],
                           help="Ollama runs locally on the group GPU. Gemini "
                                "is a hosted API with a shared daily quota.")
    llm_model = st.text_input(
        "Model name", value="qwen3:8b" if backend == "ollama" else "gemini-3.6-flash")
    retriever_name = st.selectbox(
        "Retrieval", ["dense", "hybrid", "bm25"],
        help="dense: meaning. bm25: exact words. hybrid: both, fused by rank. "
             "Measured on the eval set, dense alone scores highest.")
    k = st.slider("Excerpts retrieved", 3, 20, 8)

    try:
        index, embedder = load_index(str(INDEX_DIR), EMBED_MODEL)
        st.divider()
        st.caption(f"{len(index):,} chunks indexed")
        for repo, commit in (index.manifest.get("repo_commits") or {}).items():
            st.caption(f"{repo} @ {commit[:8]}")
        st.caption(f"built {index.manifest.get('built_at', '?')}")
    except Exception as exc:                                # noqa: BLE001
        st.error(f"Could not load the index: {exc}")
        st.stop()

    st.divider()
    st.caption("Answers are correct or partly correct about 86% of the time on "
               "a 42-question benchmark. Verify before you act.")

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try one of these**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.question = ex

question = st.text_area("Your question", value=st.session_state.question,
                        height=80, placeholder="How do I ...?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    retriever = load_retriever(index, embedder, retriever_name, str(INDEX_DIR))

    with st.spinner("Searching the corpus..."):
        chunks = retriever.search(question, k=k)

    if not chunks:
        st.warning("Nothing in the corpus matched this question.")
        st.stop()

    prompt = build_prompt(question, chunks)

    try:
        llm = get_backend(backend, model=llm_model)
        with st.spinner(f"Asking {llm_model}..."):
            answer = llm.generate(prompt)
    except LLMError as exc:
        st.error(f"{exc}")
        answer = None

    log_query(question, retriever_name, llm_model, len(chunks))

    if answer:
        st.markdown("### Answer")
        st.markdown(link_citations(answer, chunks))

    st.markdown("### What it read")
    st.caption(f"{len(chunks)} excerpts, roughly "
               f"{estimate_tokens(prompt):,} tokens of prompt")

    for i, c in enumerate(chunks, 1):
        flags = []
        if c.get("status"):
            flags.append(c["status"])
        if c.get("public_api"):
            flags.append("public API")
        tag = "  ·  " + ", ".join(flags) if flags else ""

        label = (f"{i}.  {c['repo']}/{c['path']}:{c['start_line']}"
                 f"  ·  {c['name']}  ·  score {c['score']:.3f}{tag}")
        with st.expander(label, expanded=(i == 1)):
            if c.get("status") == "deprecated":
                st.warning("This code is deprecated.")
            elif c.get("status") == "abstract":
                st.info("This is an interface to implement in a subclass, "
                        "not a function to call.")
            elif c.get("status") == "unsupported":
                st.warning("This deliberately raises in this class.")

            url = permalink(c)
            if url:
                st.markdown(f"[View on GitHub]({url})")
            st.code(c.get("context_text") or c["text"], language="python")

    with st.expander("Prompt sent to the model"):
        st.text(prompt)