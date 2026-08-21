"""The whole pipeline: question in, answer with sources out.

This is the only module that knows all three stages exist. Keeping it thin
means the CLI, the evaluation harness, and the web app in a later phase all
share one code path rather than each reassembling the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .embed import Embedder, get_embedder
from .index import VectorIndex
from .llm import LLMBackend, get_backend
from .prompt import build_prompt, estimate_tokens, permalink


@dataclass
class Answer:
    """An answer plus everything needed to check it."""

    question: str
    text: str
    sources: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    model: str = ""

    def links(self) -> list[str]:
        return [permalink(c) for c in self.sources]


class Assistant:
    """Retrieval plus generation over one index.

    Built once and reused: loading the index and the embedding model takes a
    few seconds, and doing it per question would dominate the response time.
    """

    def __init__(self, index: VectorIndex, embedder: Embedder,
                 llm: LLMBackend, k: int = 8):
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.k = k

    @classmethod
    def load(cls, index_dir: Path | str = "build",
             embed_model: str = "bge-small",
             llm_backend: str = "gemini", k: int = 8,
             **llm_kwargs) -> "Assistant":
        embedder = get_embedder(embed_model)
        # Passing the embedder makes VectorIndex check it against the manifest,
        # which catches querying an index built by a different model.
        index = VectorIndex.load(Path(index_dir), embedder)
        return cls(index, embedder, get_backend(llm_backend, **llm_kwargs), k=k)

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        """The retrieval half on its own -- useful for evaluation, which
        measures recall without spending an LLM call."""
        return self.index.search(self.embedder.embed_query(question),
                                 k=k or self.k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        chunks = self.retrieve(question, k=k)
        prompt = build_prompt(question, chunks)
        return Answer(
            question=question,
            text=self.llm.generate(prompt),
            sources=chunks,
            prompt_tokens=estimate_tokens(prompt),
            model=self.llm.model,
        )