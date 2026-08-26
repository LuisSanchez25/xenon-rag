#!/usr/bin/env python
"""Measure retrieval and answer quality against eval/questions.yaml.

Two modes, deliberately separate.

    python eval/run_eval.py --retrievers dense bm25 hybrid

Retrieval only: does the gold file appear in the top k? No LLM calls, so this
runs in seconds and costs nothing. Tune chunking, fusion and k here.

    python eval/run_eval.py --answers --backend gemini --retriever hybrid

Answer quality: actually asks the model and writes the replies to a file for
grading. One request per question, so mind the free-tier daily limit.

Refusal questions (empty gold_files) are excluded from recall -- there is no
correct file to find -- and only appear in the answer run, where what matters
is whether the model declines instead of inventing something.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml                                            # noqa: E402

from xenonrag.embed import get_embedder                # noqa: E402
from xenonrag.index import VectorIndex                 # noqa: E402
from xenonrag.retrieve import build_retriever          # noqa: E402
from xenonrag.prompt import build_prompt               # noqa: E402

import re                                              # noqa: E402

# Matches a bracketed citation like [strax/strax/context.py:300] or
# [.../peaks.py:15-40].
CITATION = re.compile(r"\[[\w./-]+\.\w+:\d+(?:-\d+)?\]")


def load_questions(path: Path) -> list[dict]:
    rows = yaml.safe_load(path.read_text())
    for r in rows:
        if "question" not in r:
            raise ValueError(f"{r.get('id')}: no 'question' key "
                             f"(keys present: {sorted(r)})")
        gf = r.get("gold_files")
        if isinstance(gf, str):
            raise ValueError(f"{r['id']}: gold_files is a string, not a list")
        r["gold_files"] = gf or []
    return rows


def hit(chunk: dict, gold: str) -> bool:
    """Does this chunk come from the gold file?

    Gold entries are "repo/path", and a trailing slash means a directory, so
    any file beneath it counts.
    """
    full = f"{chunk['repo']}/{chunk['path']}"
    gold = gold.rstrip("/")
    return full == gold or full.startswith(gold + "/")


def recall_at(hits: list[dict], gold_files: list[str], k: int,
              require_all: bool) -> bool:
    top = hits[:k]
    found = [g for g in gold_files if any(hit(c, g) for c in top)]
    return len(found) == len(gold_files) if require_all else bool(found)


def rank_of_first(hits: list[dict], gold_files: list[str]) -> int | None:
    for i, c in enumerate(hits, 1):
        if any(hit(c, g) for g in gold_files):
            return i
    return None


def run_retrieval(questions, retrievers, ks, deep):
    """Returns results[retriever][question_id] = {'ranks':..., 'recall':...}."""
    scored = [q for q in questions if q["gold_files"]]
    print(f"{len(scored)} scoreable questions "
          f"({len(questions) - len(scored)} refusal-only)\n")

    results = {}
    for r in retrievers:
        per_q = {}
        for q in scored:
            hits = r.search(q["question"], k=deep)
            per_q[q["id"]] = {
                "rank": rank_of_first(hits, q["gold_files"]),
                "recall": {k: recall_at(hits, q["gold_files"], k,
                                        q.get("require_all", False))
                           for k in ks},
                "category": q.get("category", "?"),
                "require_all": q.get("require_all", False),
            }
        results[r.name] = per_q
    return results, scored


def report(results, scored, ks):
    names = list(results)

    print(f"{'':<10}" + "".join(f"{'recall@' + str(k):>12}" for k in ks))
    for name in names:
        row = results[name]
        cells = []
        for k in ks:
            n = sum(1 for v in row.values() if v["recall"][k])
            cells.append(f"{n}/{len(row)} ({100*n/len(row):3.0f}%)")
        print(f"{name:<10}" + "".join(f"{c:>12}" for c in cells))

    print("\nrecall@5 by category")
    cats = sorted({v["category"] for v in results[names[0]].values()})
    print(f"{'':<16}" + "".join(f"{n:>10}" for n in names))
    for cat in cats:
        cells = []
        for name in names:
            rows = [v for v in results[name].values() if v["category"] == cat]
            n = sum(1 for v in rows if v["recall"][5])
            cells.append(f"{n}/{len(rows)}")
        print(f"{cat:<16}" + "".join(f"{c:>10}" for c in cells))

    if len(names) > 1:
        base, *others = names
        for other in others:
            moved = [(qid, results[base][qid]["rank"], results[other][qid]["rank"])
                     for qid in results[base]
                     if results[base][qid]["recall"][5]
                     != results[other][qid]["recall"][5]]
            if not moved:
                continue
            print(f"\nquestions where {other} differs from {base} at k=5")
            for qid, a, b in sorted(moved):
                arrow = "WON " if (b or 999) < (a or 999) else "LOST"
                print(f"  {arrow} {qid}  {base} rank {a}  ->  {other} rank {b}")


def run_answers(questions, retriever, backend_name, out_path, k,
                llm_model=None):
    from xenonrag.llm import get_backend

    llm = get_backend(backend_name, **({"model": llm_model} if llm_model else {}))
    print(f"  model: {llm.name}")
    out = []
    for i, q in enumerate(questions, 1):
        chunks = retriever.search(q["question"], k=k)
        print(f"  [{i}/{len(questions)}] {q['id']}", flush=True)
        try:
            text = llm.generate(build_prompt(q["question"], chunks))
        except Exception as exc:                        # noqa: BLE001
            text = f"<<ERROR: {exc}>>"
        cites = CITATION.findall(text)
        retrieved_paths = {f"{c['repo']}/{c['path']}" for c in chunks}
        grounded = [c for c in cites
                    if any(c.strip("[]").rsplit(":", 1)[0] == p
                           for p in retrieved_paths)]
        out.append({
            "n_citations": len(cites),
            # A citation naming a file that was not in the excerpts is
            # fabricated -- worse than no citation, since it looks checkable.
            "n_citations_grounded": len(grounded),
            "id": q["id"], "question": q["question"],
            "category": q.get("category"), "expect": q.get("expect", ""),
            "gold_files": q["gold_files"],
            "retrieved": [f"{c['repo']}/{c['path']}:{c['start_line']}"
                          for c in chunks],
            "answer": text,
            "retriever": retriever.name, "backend": llm.name,
        })
    out_path.write_text(json.dumps(out, indent=2))

    n = len(out)
    cited = sum(1 for r in out if r["n_citations"])
    clean = sum(1 for r in out
                if r["n_citations"] and r["n_citations_grounded"] == r["n_citations"])
    print(f"\ncitations: {cited}/{n} answers cite anything, "
          f"{clean}/{n} cite only retrieved files")
    print(f"wrote {n} answers to {out_path}")
    print("Grade by hand: correct / partial / wrong, and for refusal questions "
          "whether it declined instead of inventing an answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    ap.add_argument("--index", type=Path, default=Path("build"))
    ap.add_argument("--model", "--embed-model", dest="model",
                    default="bge-small",
                    help="EMBEDDING model. Must match the one the index was "
                         "built with, or VectorIndex.load will refuse.")
    ap.add_argument("--llm-model",
                    help="LLM to answer with, e.g. qwen3:8b or "
                         "gemini-3.6-flash. Defaults to the backend's own.")
    ap.add_argument("--retrievers", nargs="+", default=["dense"],
                    help="dense, bm25, hybrid, or hybrid:W to weight dense W "
                         "times bm25 (e.g. hybrid:3)")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--deep", type=int, default=50,
                    help="how far down to look when recording the gold rank")
    ap.add_argument("--answers", action="store_true",
                    help="also generate answers (costs LLM calls)")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "ollama"])
    ap.add_argument("--retriever", default="dense",
                    help="which retriever to use for the answer run")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("eval/answers.json"))
    args = ap.parse_args()

    questions = load_questions(args.questions)
    embedder = get_embedder(args.model)
    index = VectorIndex.load(args.index, embedder)

    retrievers = [build_retriever(n, index, embedder) for n in args.retrievers]
    results, scored = run_retrieval(questions, retrievers, args.ks, args.deep)
    report(results, scored, args.ks)

    if args.answers:
        r = build_retriever(args.retriever, index, embedder)
        print(f"\ngenerating answers: {r.name} + {args.backend}")
        run_answers(questions, r, args.backend, args.out, args.k,
                    llm_model=args.llm_model)


if __name__ == "__main__":
    main()