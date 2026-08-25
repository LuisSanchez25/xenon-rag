"""Turning retrieved chunks into a prompt.

Two jobs here, and both are where answer quality is actually decided.

*Budgeting.* Each chunk carries two texts: the short `text` that was embedded,
and `context_text`, which may be the whole function. Expanding all eight
retrieved chunks can produce a prompt of ~18k tokens -- fine for a hosted
model, too much for a 7B model on a 12 GB card, where anything past the context
window is silently dropped. So chunks are expanded in rank order until a budget
is reached, and lower-ranked ones fall back to their short form. The answer
degrades gradually instead of losing its tail without warning.

*Instructions.* The system prompt is what stops the model answering from its
own vague impression of "a physics analysis framework" instead of from the
excerpts. It is deliberately repetitive about that, and about saying so when
the excerpts do not contain the answer -- for a tool aimed at scientists who
will act on the reply, a confident wrong answer is worse than no answer.
"""

from __future__ import annotations

# Roughly 6k tokens of excerpts, leaving room for the question and the reply
# inside an 8k window.
CONTEXT_BUDGET_CHARS = 24_000

# No single chunk may dominate the budget. Some strax functions expand to
# 13k characters on their own.
MAX_CHUNK_CHARS = 6_000

ORG = {
    "strax": "AxFoundation", 
    "straxen": "XENONnT", 
    "xedocs": "XENONnT",
    "rframe": "XENONnT",
    }


SYSTEM = """You answer questions about the XENON analysis software stack
(strax, straxen, xedocs) for physicists who use these tools day to day.

Rules:

1. Answer ONLY from the excerpts below. Do not fall back on general knowledge
   of Python or of other analysis frameworks to fill gaps.

2. If the excerpts do not answer the question, say so plainly and say what is
   missing. Suggest where the user might look instead. Do NOT guess at
   function names, arguments, or behaviour -- a confident wrong answer is
   worse than no answer here, because the user will run it.

3. Cite the source for every specific claim, as [repo/path.py:LINE], using the
   line number given in the excerpt header.

4. If an excerpt is marked:
     - abstract    it is an interface the user is expected to implement in a
                   subclass, not a function to call. Say so.
     - deprecated  warn that it is deprecated before describing it.
     - unsupported it deliberately refuses to work in that class. Say so.

5. Show a code example ONLY by quoting one that appears in the excerpts. Never
   construct your own usage example. Plugins in this framework are not called
   directly, so an invented snippet will look plausible and not work. If no
   excerpt shows usage, say that the excerpts do not include an example.

6. If the excerpts show several implementations of the same thing -- a base
   class and its subclasses, or "vanilla" and specialised variants -- name all
   of them and explain how they relate, rather than describing only the
   highest-ranked one. Say which is the default where the excerpts make that
   clear.
   
7. Be concise. The reader is a working scientist who wants the answer, not an
   essay. Do not quote long stretches of code; quote the few lines that matter
   and describe the rest."""


def permalink(chunk: dict) -> str:
    """A GitHub link to the exact lines this chunk came from.

    Uses the commit recorded at index time, so the link keeps pointing at the
    code that was actually indexed even after the branch moves on.
    """
    org = ORG.get(chunk["repo"], "XENONnT")
    if org is None:
        return None
    return (f"https://github.com/{org}/{chunk['repo']}/blob/{chunk['commit']}/"
            f"{chunk['path']}#L{chunk['start_line']}-L{chunk['end_line']}")


def select_context(chunks: list[dict],
                   budget: int = CONTEXT_BUDGET_CHARS,
                   max_chunk: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Choose how much of each chunk to show, in retrieval-rank order.

    Returns copies with a `_body` field holding the text to put in the prompt.
    Top-ranked chunks get their full context; once the budget runs low, the
    rest fall back to the short embedded form.
    """
    out, used = [], 0
    for c in chunks:
        full = c.get("context_text") or c["text"]
        if len(full) > max_chunk:
            full = full[:max_chunk] + "\n# ... truncated ..."

        if used + len(full) <= budget:
            body = full
        else:
            body = c["text"]                      # short form always fits
        out.append({**c, "_body": body})
        used += len(body)
    return out


def format_excerpt(chunk: dict, n: int) -> str:
    """One numbered excerpt, with enough header for the model to cite it."""
    flags = []
    if chunk.get("status"):
        flags.append(chunk["status"])
    if chunk.get("public_api"):
        flags.append("public API")
    tag = f"  [{', '.join(flags)}]" if flags else ""

    header = (f"[{n}] {chunk['repo']}/{chunk['path']}:"
              f"{chunk['start_line']}-{chunk['end_line']}{tag}")
    return f"{header}\n{chunk.get('_body', chunk['text'])}"


def build_prompt(question: str, chunks: list[dict],
                 budget: int = CONTEXT_BUDGET_CHARS) -> str:
    """Assemble the full prompt sent to the model."""
    selected = select_context(chunks, budget=budget)
    excerpts = "\n\n---\n\n".join(
        format_excerpt(c, i) for i, c in enumerate(selected, 1)
    )
    return (
        f"{SYSTEM}\n\n"
        f"# Excerpts\n\n{excerpts}\n\n"
        f"# Question\n\n{question}\n\n"
        f"# Answer\n"
    )


def estimate_tokens(text: str) -> int:
    """Rough token count. Good enough for checking a prompt fits a window."""
    return len(text) // 4