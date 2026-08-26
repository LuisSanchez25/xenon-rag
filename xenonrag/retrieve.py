"""Retrieval strategies.

Three of them, behind one interface so the evaluation can swap between them:

    DenseRetriever   embedding similarity. Good at meaning, blurs exact tokens.
    BM25Retriever    classical keyword scoring. Good at exact tokens, blind to
                     paraphrase.
    HybridRetriever  both, fused by rank.

On BM25, briefly, since it is the least familiar of the three. It scores a
chunk by how many of the query's words it contains, with three corrections:
repeated words give diminishing returns, rare words count for more than common
ones, and long chunks are discounted so they cannot win just by containing more
text. In this corpus "correction" appears everywhere and so counts for almost
nothing, while "merge_without_s1" appears twice and is nearly decisive.

On fusing, the two retrievers produce scores on incomparable scales -- cosine
similarities cluster in a narrow band near 0.8, BM25 scores are unbounded and
depend on corpus statistics. Normalising them is fiddly and adds a parameter to
tune. Reciprocal rank fusion sidesteps it by discarding the scores and keeping
only the ranks: each retriever contributes 1/(k + rank) per chunk, and the
contributions add. With k=60 the top few ranks differ by under 2%, so what wins
is appearing in *both* lists rather than topping one of them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .embed import Embedder
from .index import VectorIndex


# Split identifiers at camelCase boundaries before lowering, so PeakBasics
# also matches a query saying "peak basics".
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokens for BM25.

    Underscored identifiers are emitted whole *and* split into parts, so
    `merge_without_s1` matches both a query naming it exactly and one asking
    about "merging without s1". Keeping only the parts would throw away the
    rare-token advantage that makes BM25 worth having; keeping only the whole
    would miss the paraphrase.
    """
    out: list[str] = []
    for tok in _WORD.findall(_CAMEL.sub(" ", text).lower()):
        out.append(tok)
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p)
    return out


class Retriever(ABC):
    """Anything that can rank chunks against a question."""

    name: str

    @abstractmethod
    def search(self, question: str, k: int = 8) -> list[dict]:
        """Return the top k chunks, each with a `score`, best first."""


class DenseRetriever(Retriever):
    """Embedding similarity -- what the index already does."""

    def __init__(self, index: VectorIndex, embedder: Embedder):
        self.name = "dense"
        self.index = index
        self.embedder = embedder

    def search(self, question: str, k: int = 8) -> list[dict]:
        return self.index.search(self.embedder.embed_query(question), k=k)


class BM25Retriever(Retriever):
    """Classical keyword scoring over the same chunks."""

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.name = "bm25"
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, question: str, k: int = 8) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order
                if scores[i] > 0]


class HybridRetriever(Retriever):
    """Dense and BM25, fused by reciprocal rank.

    `pool` is how deep to go in each retriever before fusing. It must be well
    above the final k: a chunk sitting at dense rank 25 can still win overall
    if BM25 ranks it first, and truncating to 8 before fusing would discard it.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 pool: int = 40, rrf_k: int = 60):
        self.name = "hybrid"
        self.dense = dense
        self.bm25 = bm25
        self.pool = pool
        self.rrf_k = rrf_k

    @staticmethod
    def _key(c: dict) -> tuple:
        return (c["repo"], c["path"], c["start_line"], c["name"])

    def search(self, question: str, k: int = 8) -> list[dict]:
        lists = [self.dense.search(question, k=self.pool),
                 self.bm25.search(question, k=self.pool)]

        scores: dict[tuple, float] = {}
        seen: dict[tuple, dict] = {}
        for ranking in lists:
            for rank, c in enumerate(ranking):
                key = self._key(c)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                seen.setdefault(key, c)

        best = sorted(scores, key=lambda key: -scores[key])[:k]
        return [{**seen[key], "score": scores[key]} for key in best]


def build_retriever(name: str, index: VectorIndex,
                    embedder: Embedder, **kwargs) -> Retriever:
    dense = DenseRetriever(index, embedder)
    if name == "dense":
        return dense
    if name == "bm25":
        return BM25Retriever(index.chunks)
    if name == "hybrid":
        return HybridRetriever(dense, BM25Retriever(index.chunks), **kwargs)
    raise ValueError(f"unknown retriever: {name!r}")