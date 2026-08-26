"""Building and querying the vector index.

The index is a directory, not a single file:

    build/
        index.faiss     the vectors, in corpus order
        chunks.jsonl    the chunks, in the same order -- row i of the index
                        corresponds to line i of this file
        manifest.json   what was indexed, with what, when

That ordering correspondence is the only thing tying the two together, so
nothing may reorder one without the other. `VectorIndex.load` checks the counts
match and refuses to open a mismatched pair.

The manifest exists because of a specific silent failure: if you build an index
with one embedding model and query it with another, FAISS will happily return
nearest neighbours. They are meaningless -- the two models place text in
unrelated coordinate systems -- but nothing errors and the answers merely get
worse. Recording the model name and checking it at load time turns that into a
loud failure.

FAISS is a similarity-search library. We use IndexFlatIP, which is exhaustive:
every query is compared against every vector. "Flat" means no approximation, so
results are exact, and "IP" means inner product, which equals cosine similarity
for the unit-length vectors our embedder produces. Approximate indexes exist for
corpora in the millions; at a few thousand chunks, exhaustive search takes
under a millisecond and there is nothing to gain.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embed import Embedder


MANIFEST = "manifest.json"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"


def build_index(chunks: list[dict], embedder: Embedder, out_dir: Path,
                repo_commits: dict[str, str] | None = None) -> dict:
    """Embed every chunk and write the index directory.

    Chunks are embedded from their `text` field -- the short form, capped to
    fit the model's context window. `context_text` is what later goes to the
    LLM, and is never embedded.
    """
    import faiss

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_documents(texts)

    if vectors.shape != (len(chunks), embedder.dim):
        raise ValueError(
            f"embedder returned {vectors.shape}, "
            f"expected {(len(chunks), embedder.dim)}"
        )

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)
    faiss.write_index(index, str(out_dir / INDEX_FILE))

    with (out_dir / CHUNKS_FILE).open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    manifest = {
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "n_chunks": len(chunks),
        "repo_commits": repo_commits or {},
        "kinds": _count(chunks, "kind"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def _count(chunks: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in chunks:
        out[str(c.get(field))] = out.get(str(c.get(field)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


class VectorIndex:
    """A loaded index, ready to answer similarity queries."""

    def __init__(self, index, chunks: list[dict], manifest: dict):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest

    @classmethod
    def load(cls, out_dir: Path, embedder: Embedder | None = None) -> "VectorIndex":
        """Open an index directory.

        If an embedder is supplied, its name and dimension are checked against
        the manifest. Skipping that check is how you get silently meaningless
        results, so pass it whenever you can.
        """
        import faiss

        out_dir = Path(out_dir)
        manifest = json.loads((out_dir / MANIFEST).read_text())
        index = faiss.read_index(str(out_dir / INDEX_FILE))

        with (out_dir / CHUNKS_FILE).open() as f:
            chunks = [json.loads(line) for line in f]

        if len(chunks) != index.ntotal:
            raise ValueError(
                f"index has {index.ntotal} vectors but chunks.jsonl has "
                f"{len(chunks)} lines -- the pair is out of sync, rebuild"
            )

        if embedder is not None:
            if embedder.name != manifest["embedding_model"]:
                raise ValueError(
                    f"index was built with {manifest['embedding_model']!r} but "
                    f"you are querying with {embedder.name!r}. The vectors are "
                    f"not comparable; rebuild the index or switch models."
                )
            if embedder.dim != manifest["dim"]:
                raise ValueError(
                    f"dimension mismatch: index {manifest['dim']}, "
                    f"embedder {embedder.dim}"
                )

        return cls(index, chunks, manifest)

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[dict]:
        """Return the k most similar chunks, each with a `score`.

        Scores are cosine similarities in [-1, 1]; higher is more similar. In
        practice anything above ~0.75 is a strong match and below ~0.5 is
        usually noise, but calibrate against your own eval set rather than
        trusting those numbers.
        """
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(q, min(k, self.index.ntotal))

        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:            # FAISS pads with -1 when k > ntotal
                continue
            out.append({**self.chunks[int(idx)], "score": float(score)})
        return out

    def __len__(self) -> int:
        return len(self.chunks)