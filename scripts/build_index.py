#!/usr/bin/env python
"""Build the search index from the cloned repositories.

    python scripts/build_index.py --repos repos --out build

Runs three stages: chunk every repo, embed every chunk, write the index. The
whole thing takes a couple of minutes on CPU for the XENON stack, and only
needs rerunning when the repos or the chunking change.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xenonrag.chunking import chunk_repo, dedupe_chunks       # noqa: E402
from xenonrag.embed import get_embedder           # noqa: E402
from xenonrag.index import build_index            # noqa: E402


DEFAULT_REPOS = ["strax", "straxen", "xedocs", "rframe"]

def head_commit(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=Path, default=Path("repos"),
                    help="directory holding the cloned repositories")
    ap.add_argument("--out", type=Path, default=Path("build"))
    ap.add_argument("--only", nargs="*", default=DEFAULT_REPOS,
                    help="which repos to index")
    ap.add_argument("--extra-docs", nargs="*", default=[], type=Path,
                    help="extra documentation directories to index. Each is "
                         "treated as its own source, named after the directory. "
                         "Use for material not in a cloned repo, e.g. a README "
                         "copied out of a private repository.")
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--device", default=None,
                    help="'cpu' or 'cuda'. CPU is fine and leaves the GPU free.")
    args = ap.parse_args()

    all_chunks: list[dict] = []
    commits: dict[str, str] = {}

    for name in args.only:
        path = args.repos / name
        if not path.is_dir():
            print(f"skipping {name}: {path} not found")
            continue
        commit = head_commit(path)
        commits[name] = commit
        chunks = chunk_repo(path, name, commit)
        print(f"{name:10s} {len(chunks):5d} chunks  @ {commit[:10]}")
        all_chunks.extend(chunks)

    for docs_dir in args.extra_docs:
        if not docs_dir.is_dir():
            print(f"skipping {docs_dir}: not a directory")
            continue
        name = docs_dir.name
        commit = head_commit(docs_dir)          # "unknown" if not a git repo
        chunks = chunk_repo(docs_dir, name, commit)
        commits[name] = commit
        all_chunks.extend(chunks)

    if not all_chunks:
        sys.exit("no chunks produced -- check --repos points at the clones")

    # Done across all repos at once, so a file vendored from one into another
    # is caught too.
    all_chunks, dropped = dedupe_chunks(all_chunks)
    if dropped:
        print(f"{'dedupe':10s} {dropped:5d} duplicate chunks removed")
    
    print(f"\ntotal {len(all_chunks)} chunks; loading {args.model} ...")
    embedder = get_embedder(args.model, device=args.device)

    manifest = build_index(all_chunks, embedder, args.out, repo_commits=commits)

    print(f"\nwrote {args.out}/")
    print(f"  model   {manifest['embedding_model']} ({manifest['dim']}d)")
    print(f"  chunks  {manifest['n_chunks']}")
    for kind, n in manifest["kinds"].items():
        print(f"    {kind:20s} {n}")

    size = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
    print(f"  size    {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()