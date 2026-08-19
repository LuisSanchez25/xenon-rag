#!/usr/bin/env python
"""Query the index and print what retrieval returns.

    python scripts/search.py "how do I load data for a run"
    python scripts/search.py "what is a plugin" -k 10 --full

No LLM involved -- this shows the raw retrieval step. Use it to judge whether
the chunks coming back could plausibly answer the question. If the right
material is not in this list, no amount of prompting will fix the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xenonrag.embed import get_embedder      # noqa: E402
from xenonrag.index import VectorIndex       # noqa: E402


ORG = {"strax": "AxFoundation", "straxen": "XENONnT", "xedocs": "XENONnT"}


def permalink(c: dict) -> str:
    org = ORG.get(c["repo"], "XENONnT")
    return (f"https://github.com/{org}/{c['repo']}/blob/{c['commit']}/"
            f"{c['path']}#L{c['start_line']}-L{c['end_line']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--index", type=Path, default=Path("build"))
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--chars", type=int, default=400)
    ap.add_argument("--full", action="store_true",
                    help="show context_text (what the LLM would see)")
    ap.add_argument("--kind", help="only show chunks of this kind")
    ap.add_argument("--links", action="store_true", help="print GitHub links")
    args = ap.parse_args()

    embedder = get_embedder(args.model)
    index = VectorIndex.load(args.index, embedder)

    hits = index.search(embedder.embed_query(args.question), k=args.k)
    if args.kind:
        hits = [h for h in hits if h["kind"] == args.kind]

    print(f"query: {args.question}")
    print(f"index: {len(index)} chunks, {index.manifest['embedding_model']}\n")

    for rank, h in enumerate(hits, 1):
        flags = []
        if h.get("status"):
            flags.append(h["status"])
        if h.get("public_api"):
            flags.append("public")
        tag = f" [{', '.join(flags)}]" if flags else ""

        print(f"{rank:2d}. {h['score']:.3f}  [{h['kind']}]{tag} "
              f"{h['repo']}/{h['path']}:{h['start_line']}  ({h['name']})")
        if args.links:
            print(f"    {permalink(h)}")

        body = h["context_text"] if args.full else h["text"]
        for line in body[:args.chars].splitlines():
            print(f"    | {line}")
        print()


if __name__ == "__main__":
    main()