"""Turning text into vectors.

A short primer, since this is the part of the project furthest from ordinary
analysis code.

An *embedding model* reads a piece of text and returns a fixed-length list of
numbers -- a vector, 384 of them for the model we use. The model is trained so
that texts meaning similar things end up as vectors pointing in similar
directions. That is the whole trick behind retrieval: to find the chunk most
relevant to a question, we embed the question, then look for the chunk whose
vector points most nearly the same way.

Two details that matter in practice:

*Normalisation.* We ask the model for unit-length vectors. Once every vector
has length 1, the dot product between two of them is exactly the cosine of the
angle between them, so "most similar" becomes "largest dot product" -- which is
a single fast matrix multiply over the whole corpus.

*The query prefix.* BGE models were trained with an instruction glued to the
front of every search query, but not to the documents being searched. Leaving
it off degrades results quietly -- nothing errors, the answers just get worse.
It lives inside the class here so no caller has to remember it.

Everything sits behind the Embedder base class so the model can be swapped
without touching the indexing or query code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Interface every embedding backend implements.

    `name` and `dim` are recorded in the index manifest so that a later query
    can check it is using the same model the index was built with.
    """

    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus chunks. Returns shape (len(texts), dim), float32."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user question. Returns shape (dim,), float32."""


class BGEEmbedder(Embedder):
    """BAAI/bge-small-en-v1.5 -- 384 dimensions, ~130 MB, CPU is fine.

    Its context window is 512 tokens. Anything longer is silently truncated,
    which is why chunking.EMBED_MAX_CHARS exists and why the chunker enforces
    it on every chunk.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

        limit = self.model.max_seq_length
        if limit is not None and limit < 512:
            # Not fatal, but worth knowing: it changes how much of each chunk
            # the model actually reads.
            print(f"note: {model_name} max_seq_length is {limit} tokens")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [self.QUERY_PREFIX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return np.asarray(vec, dtype="float32")


def get_embedder(name: str = "bge-small", **kwargs) -> Embedder:
    """Look up a backend by short name. Add new models here."""
    if name in ("bge-small", "BAAI/bge-small-en-v1.5"):
        return BGEEmbedder("BAAI/bge-small-en-v1.5", **kwargs)
    if name in ("bge-base", "BAAI/bge-base-en-v1.5"):
        return BGEEmbedder("BAAI/bge-base-en-v1.5", **kwargs)
    raise ValueError(f"unknown embedder: {name}")