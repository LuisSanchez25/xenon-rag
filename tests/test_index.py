"""Tests for xenonrag.index.

These use a StubEmbedder rather than the real model: the suite stays fast, runs
offline, and tests the plumbing rather than BAAI's model weights. That the stub
can be dropped in at all is the point of the Embedder base class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xenonrag.embed import Embedder
from xenonrag.index import VectorIndex, build_index


class StubEmbedder(Embedder):
    """Deterministic fake embeddings driven by word overlap.

    Each of the first `dim` characters of the alphabet gets a slot; a text's
    vector counts its letters and is normalised. Crude, but it makes texts
    sharing vocabulary land near each other, which is enough to test that
    retrieval returns what we expect.
    """

    def __init__(self, dim: int = 16, name: str = "stub-v1"):
        self.dim = dim
        self.name = name

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[(ord(ch) - 97) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return np.vstack([self._one(t) for t in texts]).astype("float32")

    def embed_query(self, text):
        return self._one(text)


def make_chunks(n: int = 5) -> list[dict]:
    words = ["peaklets merging", "storage backend zipfile", "plugin compute",
             "context configuration", "hitlets veto"]
    return [{
        "text": words[i % len(words)],
        "context_text": words[i % len(words)] + " (full)",
        "repo": "strax", "path": f"strax/mod{i}.py", "commit": "abc123",
        "start_line": i + 1, "end_line": i + 5,
        "kind": "function", "name": f"fn_{i}",
        "public_api": False, "status": None,
    } for i in range(n)]


@pytest.fixture
def built(tmp_path: Path):
    chunks = make_chunks()
    emb = StubEmbedder()
    manifest = build_index(chunks, emb, tmp_path / "build",
                           repo_commits={"strax": "abc123"})
    return tmp_path / "build", chunks, emb, manifest


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def test_build_writes_all_three_files(built):
    out, *_ = built
    for f in ("index.faiss", "chunks.jsonl", "manifest.json"):
        assert (out / f).exists(), f"missing {f}"


def test_manifest_records_the_model_and_commits(built):
    _, chunks, emb, manifest = built
    assert manifest["embedding_model"] == emb.name
    assert manifest["dim"] == emb.dim
    assert manifest["n_chunks"] == len(chunks)
    assert manifest["repo_commits"] == {"strax": "abc123"}
    assert "built_at" in manifest


def test_manifest_counts_kinds(built):
    _, chunks, _, manifest = built
    assert manifest["kinds"]["function"] == len(chunks)


def test_chunks_file_preserves_order(built):
    out, chunks, _, _ = built
    with (out / "chunks.jsonl").open() as f:
        written = [json.loads(l) for l in f]
    assert [c["name"] for c in written] == [c["name"] for c in chunks]


def test_build_rejects_wrong_shaped_vectors(tmp_path):
    class Broken(StubEmbedder):
        def embed_documents(self, texts):
            return np.zeros((len(texts), self.dim + 1), dtype="float32")

    with pytest.raises(ValueError, match="expected"):
        build_index(make_chunks(), Broken(), tmp_path / "b")


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def test_load_round_trips(built):
    out, chunks, emb, _ = built
    idx = VectorIndex.load(out, emb)
    assert len(idx) == len(chunks)
    assert idx.index.ntotal == len(chunks)


def test_load_rejects_a_different_model(built):
    """The silent-failure guard: different model, incomparable vectors."""
    out, *_ = built
    other = StubEmbedder(name="some-other-model")
    with pytest.raises(ValueError, match="not comparable"):
        VectorIndex.load(out, other)


def test_load_rejects_a_dimension_mismatch(built):
    out, _, emb, _ = built
    other = StubEmbedder(dim=32, name=emb.name)
    with pytest.raises(ValueError, match="dimension mismatch"):
        VectorIndex.load(out, other)


def test_load_detects_index_and_chunks_out_of_sync(built):
    """Row i of the index must be line i of chunks.jsonl."""
    out, chunks, emb, _ = built
    with (out / "chunks.jsonl").open("a") as f:
        f.write(json.dumps(chunks[0]) + "\n")
    with pytest.raises(ValueError, match="out of sync"):
        VectorIndex.load(out, emb)


def test_load_without_an_embedder_skips_validation(built):
    out, chunks, _, _ = built
    assert len(VectorIndex.load(out)) == len(chunks)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def test_search_returns_k_results_with_scores(built):
    out, _, emb, _ = built
    hits = VectorIndex.load(out, emb).search(emb.embed_query("plugin"), k=3)
    assert len(hits) == 3
    assert all("score" in h for h in hits)


def test_search_returns_results_in_descending_score_order(built):
    out, _, emb, _ = built
    hits = VectorIndex.load(out, emb).search(emb.embed_query("storage"), k=5)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_the_obviously_matching_chunk(built):
    out, _, emb, _ = built
    hits = VectorIndex.load(out, emb).search(
        emb.embed_query("storage backend zipfile"), k=1)
    assert hits[0]["text"] == "storage backend zipfile"


def test_search_results_carry_the_full_chunk_metadata(built):
    out, _, emb, _ = built
    hit = VectorIndex.load(out, emb).search(emb.embed_query("plugin"), k=1)[0]
    for field in ("context_text", "repo", "path", "start_line", "kind",
                  "name", "public_api", "status"):
        assert field in hit


def test_search_k_larger_than_corpus_is_safe(built):
    """FAISS pads with -1 when asked for more than it holds."""
    out, chunks, emb, _ = built
    hits = VectorIndex.load(out, emb).search(emb.embed_query("x"), k=999)
    assert len(hits) == len(chunks)
    assert all(h["name"] for h in hits)


def test_scores_are_cosine_similarities(built):
    """Unit vectors plus inner product means scores live in [-1, 1]."""
    out, _, emb, _ = built
    for h in VectorIndex.load(out, emb).search(emb.embed_query("plugin"), k=5):
        assert -1.01 <= h["score"] <= 1.01


def test_identical_text_scores_near_one(built):
    out, _, emb, _ = built
    hits = VectorIndex.load(out, emb).search(emb.embed_query("hitlets veto"), k=1)
    assert hits[0]["score"] > 0.99


# --------------------------------------------------------------------------
# interface contract
# --------------------------------------------------------------------------

def test_embedder_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Embedder()


def test_embed_query_returns_one_dimensional_vector():
    emb = StubEmbedder()
    v = emb.embed_query("some question")
    assert v.shape == (emb.dim,)


def test_embed_documents_returns_a_matrix():
    emb = StubEmbedder()
    m = emb.embed_documents(["a", "b", "c"])
    assert m.shape == (3, emb.dim)
    assert m.dtype == np.float32