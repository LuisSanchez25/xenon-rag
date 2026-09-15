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
import os
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
from xenonrag.serving import (QueueTimeout, RequestQueue,   # noqa: E402
                              local_backend_ready, ollama_loaded)
 
 
# The public deployment uses the committed index in build/, which covers only
# the open-source repositories. The collaboration deployment points at its own
# index, built with --extra-docs and never committed. Same code, different
# configuration -- the two deployments are not separate branches.
INDEX_DIR = Path(os.environ.get("XENONRAG_INDEX", "build"))
EMBED_MODEL = os.environ.get("XENONRAG_EMBED_MODEL", "bge-small")
QUERY_LOG = Path("logs/queries.jsonl")
 
# Ollama only exists on the group server. On a hosted deployment there is no
# local model, so the backend list and defaults change.
LOCAL_OLLAMA = bool(os.environ.get("XENONRAG_LOCAL"))
 
EXAMPLES = [
    "How do I load data for a run?",
    "What does s2_min_pmts control?",
    "How do URLConfigs work?",
    "What do I implement to write my own plugin?",
    "What does 'Cannot write to zipfiles' mean?",
]
 
CITATION = re.compile(r"\[([\w./-]+\.\w+:\d+(?:-\d+)?)\]")
 
 
# --------------------------------------------------------------------------
# loading (cached so the model and index load once per session, not per query)
# --------------------------------------------------------------------------

@st.cache_resource
def request_queue() -> RequestQueue:
    """One queue for the whole app.
 
    Streamlit runs each session in its own thread inside a single process, so
    a process-wide queue serialises every user. Ollama generates one response
    at a time regardless; the queue's job is to make the wait visible instead
    of leaving people staring at a spinner.
    """
    return RequestQueue()

 
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
 
if not LOCAL_OLLAMA:
    st.info(
        "Public demo, indexing only the open-source XENON analysis stack. "
        "Set XENONRAG_LOCAL=1 to enable the local-model backend when running "
        "on your own hardware.", icon=":material/info:")
 
with st.sidebar:
    st.header("Settings")
    backends = ["ollama", "gemini"] if LOCAL_OLLAMA else ["gemini"]
    backend = st.selectbox(
        "Model", backends,
        help=("Ollama runs locally on the group GPU. Gemini is a hosted API "
              "with a shared daily quota." if LOCAL_OLLAMA else
              "Hosted API on a shared free-tier quota. If it stops responding, "
              "the daily limit has been reached; it resets at midnight Pacific."))
    llm_model = st.text_input(
        "Model name",
        value="qwen3:8b" if backend == "ollama" else "gemini-3.6-flash")

    user_key = None
    if backend == "gemini":
        shared_key = (os.environ.get("GEMINI_API_KEY") # This is not set to work by default
                      or (st.secrets.get("GEMINI_API_KEY")
                          if hasattr(st, "secrets") else None))
        with st.expander("Use your own API key", expanded=not shared_key):
            st.caption(
                "The shared key has a daily limit across everyone using this "
                "demo. Your own key gets its own quota and is not affected by "
                "other people's use. Get one free at "
                "[Google AI Studio](https://aistudio.google.com/apikey) — no "
                "card required."
            )
            user_key = st.text_input(
                "Gemini API key", type="password", value="",
                help="Held only for this browser session, never written to "
                     "disk or logged. It is gone when you close the tab.")
            st.caption(
                ":material/lock: Sent only to Google's API. If you would rather "
                "not paste a key into a web app — a sensible default — run "
                "this locally instead; the repository is linked below.")
        if not shared_key and not user_key:
            st.warning("No shared key is configured. Enter your own above to "
                       "ask questions.")
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
    if backend == "ollama":
        ready, why = local_backend_ready()
        (st.success if ready else st.warning)(why)
        loaded = ollama_loaded()
        st.caption(f"In memory: {', '.join(loaded)}" if loaded
                   else "No model resident; the first question will load it "
                        "(a few seconds).")
    depth = request_queue().depth()
    if depth:
        st.caption(f"{depth} request{'s' if depth > 1 else ''} in progress")
 
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
 
    # Resolve the key per request rather than writing it to the environment,
    # which every session in this process would share.
    llm_kwargs = {"model": llm_model}
    if backend == "gemini":
        key = user_key or os.environ.get("GEMINI_API_KEY")
        if not key and hasattr(st, "secrets"):
            key = st.secrets.get("GEMINI_API_KEY")
        if not key:
            st.error("No API key available. Enter one in the sidebar under "
                     "\"Use your own API key\".")
            st.stop()
        llm_kwargs["api_key"] = key
 
    try:
        llm = get_backend(backend, **llm_kwargs)
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
        