"""Tests for prompt building and the end-to-end pipeline.

A StubBackend stands in for the model: the suite runs offline, costs no API
quota, and tests what we control -- what goes into the prompt -- rather than
what Google's model does with it.
"""

from __future__ import annotations

import pytest

from xenonrag.llm import LLMBackend, LLMError, _retry, get_backend
from xenonrag.prompt import (CONTEXT_BUDGET_CHARS, build_prompt,
                             estimate_tokens, format_excerpt, looks_weak,
                             permalink, select_context)


class StubBackend(LLMBackend):
    """Records the prompt it was given and returns a canned answer."""

    def __init__(self, reply: str = "stub answer"):
        self.name = "stub"
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def chunk(**kw):
    base = {
        "text": "short embedded form",
        "context_text": "the much longer full source of the function",
        "repo": "straxen", "path": "straxen/plugins/peaks.py",
        "commit": "9506b1fc16", "start_line": 15, "end_line": 40,
        "kind": "method", "name": "PeakBasics.compute",
        "public_api": False, "status": None, "score": 0.8,
        "visibility": "public",
    }
    return {**base, **kw}


# --------------------------------------------------------------------------
# permalinks
# --------------------------------------------------------------------------

def test_permalink_points_at_the_indexed_commit():
    url = permalink(chunk())
    assert "XENONnT/straxen" in url
    assert "9506b1fc16" in url
    assert url.endswith("#L15-L40")


def test_no_permalink_for_an_unknown_repo():
    """A private repo has no public URL; guessing one produces a 404."""
    assert permalink(chunk(repo="corrections")) is None


def test_no_permalink_for_internal_chunks():
    assert permalink(chunk(visibility="internal")) is None


def test_internal_chunks_are_flagged_to_the_model():
    out = format_excerpt({**chunk(visibility="internal"), "_body": "b"}, 1)
    assert "internal doc" in out


def test_permalink_uses_the_right_org_per_repo():
    assert "AxFoundation/strax" in permalink(chunk(repo="strax"))
    assert "XENONnT/xedocs" in permalink(chunk(repo="xedocs"))


# --------------------------------------------------------------------------
# context budgeting
# --------------------------------------------------------------------------

def test_small_chunks_all_get_full_context():
    sel = select_context([chunk() for _ in range(3)])
    assert all(c["_body"] == c["context_text"] for c in sel)


def test_budget_falls_back_to_the_short_form():
    """Top-ranked chunks keep full context; the tail degrades to short."""
    big = chunk(context_text="x" * 5000)
    sel = select_context([big] * 10, budget=12_000)
    assert sel[0]["_body"] == big["context_text"]
    assert sel[-1]["_body"] == big["text"]


def test_budget_is_respected():
    sel = select_context([chunk(context_text="y" * 5000)] * 10, budget=12_000)
    expanded = sum(len(c["_body"]) for c in sel
                   if c["_body"] == c["context_text"])
    assert expanded <= 12_000


def test_one_huge_chunk_cannot_dominate():
    """Some strax functions expand to 13k characters on their own."""
    sel = select_context([chunk(context_text="z" * 20_000)], max_chunk=6_000)
    assert len(sel[0]["_body"]) <= 6_100
    assert "truncated" in sel[0]["_body"]


def test_every_chunk_still_appears_even_over_budget():
    sel = select_context([chunk(context_text="q" * 9000)] * 20, budget=5_000)
    assert len(sel) == 20
    assert all(c["_body"] for c in sel)


def test_chunk_without_context_text_falls_back_to_text():
    c = chunk()
    del c["context_text"]
    assert select_context([c])[0]["_body"] == c["text"]


# --------------------------------------------------------------------------
# excerpt formatting
# --------------------------------------------------------------------------

def test_excerpt_header_is_literally_the_citation():
    """Regression: heading excerpts "[1] path:15-40" while asking for
    "[path:LINE]" made models emit "[1/path:15-40]". The header is now the
    citation, so copying it verbatim is correct."""
    out = format_excerpt({**chunk(), "_body": "body"}, 3)
    first = out.split("\n")[0]
    assert "[straxen/straxen/plugins/peaks.py:15-40]" in first
    assert "[3]" not in first


def test_excerpt_flags_do_not_use_square_brackets():
    """Square brackets are reserved for citations; anything else in them
    invites the model to cite the flag."""
    out = format_excerpt({**chunk(status="deprecated"), "_body": "b"}, 1)
    first = out.split("\n")[0]
    assert "(deprecated)" in first
    assert first.count("[") == 1


def test_status_is_surfaced_to_the_model():
    for status in ("abstract", "deprecated", "unsupported"):
        out = format_excerpt({**chunk(status=status), "_body": "b"}, 1)
        assert status in out


def test_public_api_is_surfaced():
    out = format_excerpt({**chunk(public_api=True), "_body": "b"}, 1)
    assert "public API" in out


def test_ordinary_chunk_has_no_flag_clutter():
    first = format_excerpt({**chunk(), "_body": "b"}, 1).split("\n")[0]
    assert "(" not in first


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------

def test_prompt_contains_question_and_excerpts():
    p = build_prompt("how do I merge peaklets?", [chunk()])
    assert "how do I merge peaklets?" in p
    assert "the much longer full source" in p


def test_prompt_instructs_the_model_not_to_guess():
    p = build_prompt("q", [chunk()])
    assert "ONLY from the excerpts" in p
    assert "Do NOT guess" in p


def test_prompt_asks_for_citations():
    p = build_prompt("q", [chunk()])
    assert "copying the bracketed string" in p
    assert "Source:" in p


def test_format_reminder_comes_after_the_question():
    """Small models follow a rule stated once near the top unevenly. A worked
    example placed last is followed far more reliably -- qwen2.5-coder:7b cited
    nothing with the rule alone."""
    p = build_prompt("how do peaklets work?", [chunk()])
    assert p.index("Required format") > p.index("how do peaklets work?")
    assert p.rstrip().endswith("# Answer")


def test_format_reminder_shows_a_worked_example():
    p = build_prompt("q", [chunk()])
    assert "peaklet_classification_vanilla.py:73]" in p
    assert "not acceptable" in p


def test_prompt_explains_the_status_flags():
    p = build_prompt("q", [chunk()])
    for word in ("abstract", "deprecated", "unsupported"):
        assert word in p


def test_prompt_asks_for_synthesis_across_variants():
    """Retrieval returned both PeakletClassificationVanilla and its SOM
    subclass, including the super() call, and the model still answered from
    the top-ranked chunk alone. The instruction is what was missing."""
    p = build_prompt("q", [chunk()])
    assert "super()" in p
    assert "Inherits from" in p
    assert "subclasses" in p


def test_prompt_forbids_inventing_code_examples():
    """Ollama produced `PeakletClassificationVanilla().compute(peaklets)`,
    which is not how strax plugins are used and appeared in no excerpt."""
    p = build_prompt("q", [chunk()])
    assert "construct your own usage example" in p
    assert "ONLY by quoting one that appears in the excerpts" in p


def test_prompt_forbids_unsupported_default_claims():
    """Both models asserted vanilla was the default. Which plugin a context
    registers lives in contexts.py, which this query never retrieves."""
    p = build_prompt("q", [chunk()])
    assert "default" in p and "unless an excerpt says so" in p


def test_prompt_discourages_long_code_quotes():
    assert "Do not reproduce long stretches of code" in build_prompt("q", [chunk()])


def test_prompt_asks_for_mechanism_not_just_outputs():
    """Regression: combining "be concise" with "don't quote code" produced a
    199-character answer that listed the output categories and never described
    how classification actually happens."""
    p = build_prompt("q", [chunk()])
    assert "the actual" in p and "mechanism" in p
    assert "Cut padding, not substance" in p


def test_excerpts_appear_in_rank_order():
    p = build_prompt("q", [chunk(start_line=i * 10 + 1) for i in range(3)])
    assert p.index(":1-40]") < p.index(":11-40]") < p.index(":21-40]")


def test_refusal_template_is_shown_not_just_described():
    """Gemini refused reliably, the 7B did not. A worked example in the last
    position is followed far more often than a rule stated once near the top."""
    p = build_prompt("q", [chunk()])
    assert "the whole reply is a refusal" in p
    assert "Do not pad a refusal" in p


def test_weak_retrieval_is_flagged_to_the_model():
    p = build_prompt("q", [chunk(score=0.42)])
    assert "nearest available" in p


def test_strong_retrieval_adds_no_note():
    assert "nearest available" not in build_prompt("q", [chunk(score=0.83)])


def test_low_confidence_can_be_forced():
    """Fused rankings score on a different scale, so the caller decides."""
    p = build_prompt("q", [chunk(score=0.031)], low_confidence=False)
    assert "nearest available" not in p


def test_fused_scores_do_not_trigger_the_note_by_accident():
    """RRF scores land around 0.03; without a floor every hybrid query would
    be flagged as weak retrieval."""
    assert not looks_weak([chunk(score=0.031)])
    assert looks_weak([chunk(score=0.42)])
    assert not looks_weak([chunk(score=0.83)])


def test_empty_retrieval_is_weak():
    assert looks_weak([])


def test_prompt_with_no_chunks_still_builds():
    """Retrieval can legitimately return nothing; that must not crash."""
    p = build_prompt("q", [])
    assert "q" in p


def test_estimate_tokens_is_in_the_right_ballpark():
    assert 200 < estimate_tokens("word " * 200) < 400


# --------------------------------------------------------------------------
# backend contract
# --------------------------------------------------------------------------

def test_backend_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMBackend()


def test_gemini_accepts_a_key_as_an_argument():
    """os.environ is process-wide. In a multi-user app every session shares one
    process, so a key written there would be used by everyone's requests."""
    import inspect
    from xenonrag.llm import GeminiBackend
    assert "api_key" in inspect.signature(GeminiBackend.__init__).parameters
    src = inspect.getsource(GeminiBackend.__init__)
    assert "api_key or os.environ" in src, "argument must take precedence"


def test_app_never_writes_the_key_to_the_environment():
    """Regression guard for the leak above."""
    from pathlib import Path
    app = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    if app.exists():
        assert 'os.environ["GEMINI_API_KEY"] =' not in app.read_text()


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("gpt-9")


def test_retry_backs_off_on_rate_limits():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert _retry(flaky, base_delay=0) == "ok"
    assert len(calls) == 3


def test_retry_does_not_mask_other_errors():
    """Retrying a bad API key just wastes time."""
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("401 invalid api key")

    with pytest.raises(RuntimeError, match="invalid api key"):
        _retry(broken, base_delay=0)
    assert len(calls) == 1


def test_retry_gives_up_eventually():
    def always():
        raise RuntimeError("429 too many requests")

    with pytest.raises(RuntimeError):
        _retry(always, attempts=3, base_delay=0)


# --------------------------------------------------------------------------
# end to end, with a stub model
# --------------------------------------------------------------------------

def test_assistant_passes_retrieved_chunks_to_the_model():
    from xenonrag.answer import Assistant

    class FakeIndex:
        def search(self, v, k=8):
            return [chunk(name="found_this")]

    class FakeEmbedder:
        name, dim = "stub", 4
        def embed_query(self, t): return [0.0] * 4
        def embed_documents(self, ts): return [[0.0] * 4 for _ in ts]

    stub = StubBackend("here is the answer")
    a = Assistant(FakeIndex(), FakeEmbedder(), stub).ask("my question")

    assert a.text == "here is the answer"
    assert a.model == "stub"
    assert a.prompt_tokens > 0
    assert len(a.sources) == 1
    assert "my question" in stub.prompts[0]
    assert a.links()[0].startswith("https://github.com/XENONnT/straxen")