"""Tests for retrieval strategies.

Uses the StubEmbedder from test_index so the suite stays offline. BM25 is
real -- it needs no model, so these test the actual scoring.
"""

from __future__ import annotations

import pytest

from xenonrag.index import VectorIndex, build_index
from xenonrag.retrieve import (BM25Retriever, DenseRetriever, HybridRetriever,
                               build_retriever, tokenize)
from test_index import StubEmbedder


CORPUS = [
    "File: straxen/plugins/merged_s2s.py\nClass: MergedS2s\n"
    "Configuration options\n\nmerge_without_s1 = strax.Option('merge_without_s1', "
    "default=True, help='If true, S1s are ignored during merging.')",

    "File: straxen/plugins/merged_s2s.py\nClass: MergedS2s\n"
    "Merge together peaklets if peak finding favours that.",

    "File: strax/processing/peak_merging.py\nFunction: merge_peaks\n"
    "Merge peaks into a single peak, combining small signals into bigger ones.",

    "File: strax/storage/zipfiles.py\nClass: ZipDirectory\n"
    "Method: write_run_metadata\n\nraise NotImplementedError('Cannot write to zipfiles')",

    "File: straxen/plugins/records/records.py\nFunction: to_pe\n"
    "Convert PMT pulse area to photoelectrons using the gain model.",
]


def chunks():
    return [{
        "text": t, "context_text": t, "repo": "demo",
        "path": t.split("\n")[0].replace("File: ", ""),
        "commit": "abc", "start_line": i + 1, "end_line": i + 2,
        "kind": "prose", "name": f"c{i}", "public_api": False,
        "status": None, "visibility": "public",
    } for i, t in enumerate(CORPUS)]


@pytest.fixture
def retrievers(tmp_path):
    cs = chunks()
    emb = StubEmbedder(dim=26)
    build_index(cs, emb, tmp_path / "b")
    idx = VectorIndex.load(tmp_path / "b", emb)
    dense = DenseRetriever(idx, emb)
    bm25 = BM25Retriever(cs)
    return dense, bm25, HybridRetriever(dense, bm25, pool=5)


# --------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------

def test_underscored_identifier_is_kept_whole_and_split():
    """Whole for exact-match power, split so a paraphrase still matches."""
    toks = tokenize("merge_without_s1")
    assert "merge_without_s1" in toks
    assert {"merge", "without", "s1"} <= set(toks)


def test_camel_case_is_split():
    toks = tokenize("PeakBasics")
    assert "peak" in toks and "basics" in toks


def test_acronym_followed_by_word_splits():
    assert "config" in tokenize("URLConfig")


def test_punctuation_is_dropped():
    assert tokenize("st.get_array(run_id)") == [
        "st", "get_array", "get", "array", "run_id", "run", "id"]


def test_tokenize_is_case_insensitive():
    assert tokenize("To_PE") == tokenize("to_pe")


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def test_bm25_finds_an_exact_identifier(retrievers):
    _, bm25, _ = retrievers
    top = bm25.search("merge_without_s1", k=1)
    assert "merge_without_s1" in top[0]["text"]


def test_bm25_finds_an_error_string(retrievers):
    _, bm25, _ = retrievers
    top = bm25.search("Cannot write to zipfiles", k=1)
    assert "zipfiles" in top[0]["path"]


def test_bm25_returns_nothing_when_no_word_matches(retrievers):
    """Its blind spot: no literal overlap means no result at all."""
    _, bm25, _ = retrievers
    assert bm25.search("quantum chromodynamics lattice", k=5) == []


def test_bm25_scores_descend(retrievers):
    _, bm25, _ = retrievers
    scores = [h["score"] for h in bm25.search("merge peaks", k=5)]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def test_hybrid_returns_k_results(retrievers):
    _, _, hybrid = retrievers
    assert len(hybrid.search("merge peaklets", k=3)) == 3


def test_hybrid_never_duplicates_a_chunk(retrievers):
    """A chunk in both lists must be fused, not returned twice."""
    _, _, hybrid = retrievers
    hits = hybrid.search("merge_without_s1 merging", k=5)
    keys = [(h["path"], h["start_line"]) for h in hits]
    assert len(keys) == len(set(keys))


def test_rrf_rewards_appearing_in_both_lists():
    """A chunk mediocre in both beats one that tops a single list.

    This is the whole point of the k=60 constant: it flattens the top of the
    curve so consensus outweighs one retriever's enthusiasm.
    """
    class Fake:
        def __init__(self, out): self.out = out; self.name = "fake"
        def search(self, q, k=8): return self.out

    def c(name):
        return {"repo": "r", "path": f"{name}.py", "start_line": 1,
                "name": name, "text": name}

    both = c("both")          # rank 3 and rank 4
    only = c("only")          # rank 1, absent from the other list
    a = Fake([c("x"), c("y"), both, c("z")])
    b = Fake([only, c("p"), c("q"), both])

    hits = HybridRetriever(a, b, pool=10).search("q", k=4)
    # "both" is rank 3 in one list and rank 4 in the other, yet beats chunks
    # that top a single list: 1/63 + 1/64 > 1/61.
    assert hits[0]["name"] == "both"
    # The rank-1 chunks tie with each other at 1/61, so their order is
    # arbitrary -- only their being beaten by "both" is meaningful.
    assert {h["name"] for h in hits[1:3]} == {"x", "only"}


def test_hybrid_recovers_a_chunk_dense_ranked_low(retrievers):
    """The motivating case: an exact identifier buried by dense retrieval."""
    dense, _, hybrid = retrievers
    q = "merge_without_s1"
    d = [h["path"] + str(h["start_line"]) for h in dense.search(q, k=5)]
    h = [x["path"] + str(x["start_line"]) for x in hybrid.search(q, k=5)]
    target = next(c["path"] + str(c["start_line"]) for c in chunks()
                  if "merge_without_s1 = strax.Option" in c["text"])
    assert h.index(target) <= d.index(target)


def test_pool_must_exceed_k_to_be_useful(retrievers):
    """Truncating each list to k before fusing throws away exactly the chunks
    fusion exists to rescue."""
    dense, bm25, _ = retrievers
    shallow = HybridRetriever(dense, bm25, pool=1).search("merge_without_s1", k=5)
    deep = HybridRetriever(dense, bm25, pool=5).search("merge_without_s1", k=5)
    assert len(deep) >= len(shallow)


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def test_build_retriever_names(tmp_path):
    cs = chunks()
    emb = StubEmbedder(dim=26)
    build_index(cs, emb, tmp_path / "b")
    idx = VectorIndex.load(tmp_path / "b", emb)
    for n in ("dense", "bm25", "hybrid"):
        assert build_retriever(n, idx, emb).name == n


def test_unknown_retriever_is_rejected(tmp_path):
    cs = chunks()
    emb = StubEmbedder(dim=26)
    build_index(cs, emb, tmp_path / "b")
    idx = VectorIndex.load(tmp_path / "b", emb)
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("magic", idx, emb)