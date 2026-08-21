#!/usr/bin/env python
"""Ask a question and get an answer with citations.

    python scripts/ask.py "how do I add a new correction to xedocs?"
    python scripts/ask.py "what does MergedS2s depend on?" --sources
    python scripts/ask.py "..." --backend ollama

If an answer looks wrong, run the same question through scripts/search.py
first. That shows retrieval alone, which tells you whether the right material
was even found -- if it was not, no prompt change will fix the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xenonrag.answer import Assistant          # noqa: E402
from xenonrag.llm import LLMError              # noqa: E402
from xenonrag.prompt import build_prompt       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=8, help="chunks to retrieve")
    ap.add_argument("--index", type=Path, default=Path("build"))
    ap.add_argument("--backend", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--model", help="override the backend's default model")
    ap.add_argument("--sources", action="store_true",
                    help="list the excerpts used, with GitHub links")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the prompt instead of calling the model")
    args = ap.parse_args()

    llm_kwargs = {"model": args.model} if args.model else {}

    try:
        assistant = Assistant.load(args.index, llm_backend=args.backend,
                                   k=args.k, **llm_kwargs)
    except LLMError as exc:
        sys.exit(f"backend error: {exc}")

    if args.show_prompt:
        chunks = assistant.retrieve(args.question)
        print(build_prompt(args.question, chunks))
        return

    try:
        answer = assistant.ask(args.question)
    except LLMError as exc:
        sys.exit(f"generation failed: {exc}")

    print(answer.text)

    if args.sources:
        print("\n" + "-" * 70)
        print(f"{len(answer.sources)} excerpts, ~{answer.prompt_tokens} prompt "
              f"tokens, {answer.model}\n")
        for i, (c, url) in enumerate(zip(answer.sources, answer.links()), 1):
            flags = f" [{c['status']}]" if c.get("status") else ""
            print(f"[{i}] {c['score']:.3f} {c['repo']}/{c['path']}:"
                  f"{c['start_line']}{flags}")
            print(f"    {url}")


if __name__ == "__main__":
    main()