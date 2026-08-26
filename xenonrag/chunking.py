"""Chunking for the XENON RAG corpus.

Two strategies:

Code (.py) — syntax-aware, using the ast module.
  * module docstring          -> one chunk
  * top-level function        -> one chunk
  * class                     -> a "class card" (docstring, base classes,
                                 class-level assignments, method signatures)
                                 plus one chunk per method
  Method bodies never appear inside the class card, so nothing is duplicated.

Prose (.md/.rst) — split on headings, then on paragraphs if a section is long.
  Every chunk is prefixed with the document title and heading path so it
  still makes sense when retrieved in isolation.

Each chunk is a dict with the text plus enough metadata to build a GitHub
permalink and to filter or debug later.
"""

from __future__ import annotations

import ast
import json
import os
import re
import hashlib
from pathlib import Path

# Embedding model limit. bge-small-en-v1.5 handles 512 tokens; code runs
# about 3-3.5 chars/token, so this leaves headroom.
EMBED_MAX_CHARS = 1500
OVERLAP_CHARS = 200

# Functions longer than this become a single summary chunk rather than
# N mediocre parts.
SUMMARY_THRESHOLD = 4000
SUMMARY_BODY_LINES = 30

PROSE_MAX_CHARS = 1200
PROSE_OVERLAP_CHARS = 150

MIN_CHUNK_CHARS = 80

SKIP_DIR_PARTS = {
    ".git", "__pycache__", "tests", "test", "_build",
    ".github", "build", "dist", ".eggs", "node_modules", ".ipynb_checkpoints",
}

SKIP_FILENAMES = {
    "HISTORY.md", "CHANGELOG.md", "CHANGES.md", "CHANGELOG.rst",
    "AUTHORS.md", "CONTRIBUTORS.md", "CODE_OF_CONDUCT.md",
    "setup.py", "conftest.py", "versioneer.py", "_version.py",
}

# A prose chunk that is mostly Sphinx directives carries no information --
# it tells Sphinx to render docstrings we already index from source.
AUTODOC_DIRECTIVE = re.compile(r"^\.\.\s+auto(?:module|class|function)::", re.M)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _tail_lines(text: str, max_chars: int) -> str:
    """Last whole lines of `text`, up to max_chars.

    Go thorugh the lines in reverse order, so we can stop at the first line 
    that would exceed the limit.

    Used for overlap so continuation chunks start at a line boundary rather
    than mid-token.
    """
    out, total = [], 0
    for line in reversed(text.splitlines()):
        cost = len(line) + (1 if out else 0)
        # No "and out" escape hatch here: a single line longer than the budget
        # must yield no overlap at all. Returning it anyway (as an earlier
        # version did) prepends a whole paragraph to the next chunk, pushing it
        # over the size cap and duplicating text across two chunks.
        if total + cost > max_chars:
            break
        out.append(line)
        total += cost
    return "\n".join(reversed(out))


def _truncate_lines(text: str, max_chars: int) -> str:
    """First whole lines of `text`, up to max_chars.
    
    Used for truncating a chunk to fit the embedding size limit.
    
    """
    if len(text) <= max_chars:
        return text
    out, total = [], 0
    for line in text.splitlines(): # Same as before but not in reverse order
        cost = len(line) + (1 if out else 0)
        if total + cost > max_chars and out:
            break
        out.append(line)
        total += cost
    return "\n".join(out)


def _mk(text, *, repo, path, commit, start_line, end_line, kind, name,
        context_text=None, public_api=False, status = None):
    """Build a chunk record. Keep this the single place chunks are created.
    
    Enforces the embedding size cap in one place: if `text` is too long it
    becomes the prompt context and the embedded form is truncated.
    """
    text = text.strip()
    context_text = (context_text or text).strip()

    if len(text) > EMBED_MAX_CHARS:
        text = _truncate_lines(text, EMBED_MAX_CHARS)

    return {
        "text": text,
        "context_text": context_text,
        "repo": repo,
        "path": path,
        "commit": commit,
        "start_line": start_line,
        "end_line": end_line,
        "kind": kind,
        "name": name,
        "public_api": public_api,
        "status": status,
    }


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on blank lines, keeping a little overlap between pieces.

    Splitting on blank lines rather than a hard character count keeps
    logical blocks intact more often than not.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    pieces, current = [], ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = _tail_lines(current, overlap) + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        pieces.append(current)

    # A single paragraph can still exceed the cap. Fall back to line-aligned
    # hard cuts rather than slicing mid-token.
    out = []
    for piece in pieces:
        while len(piece) > max_chars:
            head = _truncate_lines(piece, max_chars)
            if len(head) > max_chars or not head:   # one enormous line
                head = piece[:max_chars]
            out.append(head)
            piece = piece[len(head):].lstrip("\n")
        if piece.strip():
            out.append(piece)
    return out


def _end_line(node, default):
    return getattr(node, "end_lineno", None) or default


def should_skip(rel_path: Path) -> bool:
    if rel_path.name in SKIP_FILENAMES:
        return True
    return any(part in SKIP_DIR_PARTS for part in rel_path.parts)


# --------------------------------------------------------------------------
# code chunking
# --------------------------------------------------------------------------
# A signature longer than this is truncated. Functions with many parameters
# and long default values (straxen's context builders list a dozen filesystem
# paths inline) would otherwise fill the whole embedding window with defaults
# and push the docstring out.
MAX_SIGNATURE_CHARS = 300


DEPRECATION_WORDS = ("deprecat", "will be removed", "obsolete",
                     "does not work with", "use ... instead")


def _signature(node, max_chars: int = MAX_SIGNATURE_CHARS) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    if len(args) > max_chars:
        # Cut at a parameter boundary so the result still reads as a signature.
        cut = args.rfind(", ", 0, max_chars)
        args = (args[:cut] if cut > 0 else args[:max_chars]) + ", ..."
    return f"{prefix} {node.name}({args})"


def _is_config(node) -> bool:
    """True for `name = strax.Option(...)` / `straxen.URLConfig(...)` style."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    if node.value is None or not isinstance(node.value, ast.Call):
        return False
    try:
        fn = ast.unparse(node.value.func)
    except Exception:
        return False
    return "Option" in fn or "Config" in fn


def _target_name(node) -> str:
    if isinstance(node, ast.AnnAssign):
        return ast.unparse(node.target)
    return ", ".join(ast.unparse(t) for t in node.targets)


def _split_class_body(cls) -> tuple[list, list]:
    """Separate plain class attributes from config-option declarations."""
    plain, configs = [], []
    for node in cls.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            (configs if _is_config(node) else plain).append(node)
    return plain, configs


def _class_card(cls, src: str, rel: str) -> str:
    """Class docstring + bases + class-level assignments + method signatures.

    For a straxen plugin this is the chunk that answers 'what does this
    plugin do, what does it depend on, what config does it take' — which is
    most of what people ask. Method bodies are deliberately excluded; they
    become their own chunks.
    """
    parts = [f"File: {rel}", f"Class: {cls.name}"]

    bases = [ast.unparse(b) for b in cls.bases]
    if bases:
        parts.append(f"Inherits from: {', '.join(bases)}")

    doc = ast.get_docstring(cls)
    if doc:
        parts.append(f"\n{doc}")

    plain, configs = _split_class_body(cls)

    if plain:
        segs = [ast.get_source_segment(src, n) for n in plain]
        body = "\n".join(s for s in segs if s)
        parts.append("\nClass attributes:\n" + body)

    if configs:
        names = [_target_name(n) for n in configs]
        parts.append(f"\nConfiguration options ({len(names)}):\n"
                     + ", ".join(names))

    methods = [n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if methods:
        parts.append("\nMethods:\n" + "\n".join(_signature(m) for m in methods))

    return "\n".join(parts)


def _class_line(parent: str | None, parent_bases: list[str] | None) -> list[str]:
    """The `Class:` / `Inherits from:` lines for a method chunk.
 
    A method retrieved on its own is orphaned from its class hierarchy.
    straxen's PeakletClassificationHighEnergy.compute is nothing but
    `return super().compute(peaklets_he)` -- without the base class named in
    the header, neither the embedding nor the model has any way to find out
    what it actually does.
    """
    if not parent:
        return []
    lines = [f"Class: {parent}"]
    if parent_bases:
        lines.append(f"Inherits from: {', '.join(parent_bases)}")
    return lines


def _continuation_header(node, rel: str, parent: str | None = None,
                         parent_bases: list[str] | None = None) -> str:
    """Header for parts 2+, repeating signature and docstring so every piece
    is self-describing when retrieved alone."""
    lines = [f"File: {rel}"] + _class_line(parent, parent_bases)
    lines.append(f"{'Method' if parent else 'Function'}: {node.name}")
    # Docstring first: it carries the meaning, and if the embedding cap has to
    # cut something it should cut the parameter list.
    doc = ast.get_docstring(node)
    if doc:
        first = doc.strip().split("\n\n")[0]
        lines.append(f"\n{first[:400]}")
    lines.append(f"\nSignature: {_signature(node)}")
    return "\n".join(lines) + "\n\n"


def _body_opening(node, seg: str, n_lines: int) -> str:
    """The first few lines of real code, skipping signature and docstring.

    Taking the first N lines of the source instead would, for a function with
    a large parameter list, simply repeat the signature.
    """
    stmts = _strip_docstring(node.body)
    if not stmts:
        return ""
    offset = max(0, stmts[0].lineno - node.lineno)
    return "\n".join(seg.splitlines()[offset:offset + n_lines])


def _summary_text(node, seg: str, rel: str, parent: str | None = None,
                  parent_bases: list[str] | None = None) -> str:
    """One self-contained chunk for a very long function: signature, full
    docstring, and the opening of the body."""
    lines = [f"File: {rel}"]
    if parent:
        lines.append(f"Class: {parent}")
    lines.append(f"{'Method' if parent else 'Function'}: {node.name}")
    # Docstring before signature, for the same reason as in
    # _continuation_header: the description must survive truncation.
    doc = ast.get_docstring(node)
    if doc:
        lines.append(f'\n"""{doc.strip()}"""')
 
    lines.append(f"\n{_signature(node)}")
    lines.append("\n" + _body_opening(node, seg, SUMMARY_BODY_LINES))
    lines.append(f"\n# ... continues to line {_end_line(node, node.lineno)} ...")
    return "\n".join(lines)


def _strip_docstring(body: list) -> list:
    """Function body without its leading docstring.
 
    Note a bare call like `log.warning(...)` is also an ast.Expr, so we can
    only drop the *first* statement, and only if it is a string literal.
    """
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _raises_not_implemented(node) -> bool:
    """Does the body raise NotImplementedError directly?

    Checked structurally rather than by substring, so a mention of the
    exception in a docstring or comment doesn't count.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Raise) or sub.exc is None:
            continue
        exc = sub.exc
        target = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(target, ast.Name) and target.id == "NotImplementedError":
            return True
    return False


def _classify(node, seg: str) -> str | None:
    """Tag a function as abstract, unsupported, deprecated, or nothing.
 
    abstract     a stub the user is expected to override (Plugin.compute)
    unsupported  a real method that refuses on purpose (ZipDirectory.remove)
    deprecated   kept only for backwards compatibility
    """
    doc = (ast.get_docstring(node) or "").lower()
    head = seg[:800].lower()
    if any(w in head or w in doc for w in DEPRECATION_WORDS):
        return "deprecated"
    if not _raises_not_implemented(node):
        return None
    # If the raise is essentially the whole body, this is an interface stub.
    real = _strip_docstring(node.body)
    return "abstract" if len(real) <= 1 else "unsupported"


def _emit_callable(node, src, *, repo, rel, commit, chunks, parent=None,
                   parent_bases=None, public_api=False):
    """Emit chunk(s) for one function or method."""
    seg = ast.get_source_segment(src, node)
    if not seg:
        return
 
    status = _classify(node, seg)
    kind = "method" if parent else "function"
    qual = f"{parent}.{node.name}" if parent else node.name
 
    # A function is summarised either because it is long, or because its
    # signature is so large that the raw source would embed as a wall of
    # parameter defaults with the docstring truncated off the end.
    try:
        raw_sig_len = len(ast.unparse(node.args))
    except Exception:
        raw_sig_len = 0
 
    wide_and_truncated = (raw_sig_len > MAX_SIGNATURE_CHARS
                          and len(seg) > EMBED_MAX_CHARS)
    if len(seg) > SUMMARY_THRESHOLD or wide_and_truncated:
        chunks.append(_mk(
            _summary_text(node, seg, rel, parent, parent_bases),
            context_text=seg,
            repo=repo, path=rel, commit=commit,
            start_line=node.lineno, end_line=_end_line(node, node.lineno),
            kind=f"{kind}_summary", name=qual, public_api=public_api,
            status=status,
        ))
        return
 
    base_header = [f"File: {rel}"] + _class_line(parent, parent_bases)
    base_header.append(f"{'Method' if parent else 'Function'}: {node.name}")
    first_header = "\n".join(base_header) + "\n\n"
 
    pieces = _split_long(seg, EMBED_MAX_CHARS, OVERLAP_CHARS)
    for i, piece in enumerate(pieces):
        header = (first_header if i == 0
                  else _continuation_header(node, rel, parent, parent_bases))
        suffix = "" if i == 0 else f" (part {i + 1})"
        chunks.append(_mk(
            header + piece,
            context_text=(first_header + seg) if len(pieces) > 1 else None,
            repo=repo, path=rel, commit=commit,
            start_line=node.lineno, end_line=_end_line(node, node.lineno),
            kind=kind, name=qual + suffix, public_api=public_api,
            status=status,
        ))


def chunk_python(path: Path, repo: str, commit: str, root: Path,
                 public_modules: set[str] | None = None) -> list[dict]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    try:
        # filename= so a SyntaxWarning from third-party source names the file
        # rather than reporting "<unknown>".
        tree = ast.parse(src, filename=rel)
    except SyntaxError:
        return []
 
    module = rel.replace("/", ".").removesuffix(".py").removesuffix(".__init__")
    public = bool(public_modules) and any(
        module.endswith(m) or m.endswith(module) for m in public_modules
    )
 
    chunks: list[dict] = []
 
    mod_doc = ast.get_docstring(tree)
    if mod_doc and len(mod_doc.strip()) > 40:
        chunks.append(_mk(
            f"File: {rel}\nModule overview\n\n{mod_doc}",
            repo=repo, path=rel, commit=commit,
            start_line=1, end_line=1,
            kind="module_docstring", name=Path(rel).stem, public_api=public,
        ))
 
    for node in tree.body:
 
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _emit_callable(node, src, repo=repo, rel=rel, commit=commit,
                           chunks=chunks, public_api=public)
 
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            full_card = _class_card(node, src, rel)
            chunks.append(_mk(
                full_card, context_text=full_card,
                repo=repo, path=rel, commit=commit,
                start_line=node.lineno, end_line=_end_line(node, node.lineno),
                kind="class_card", name=node.name, public_api=public,
            ))
 
            _, configs = _split_class_body(node)
            for n in configs:
                seg = ast.get_source_segment(src, n)
                if not seg:
                    continue
                name = _target_name(n)
                header = (f"File: {rel}\nClass: {node.name}\n"
                          f"Configuration option: {name}\n\n")
                chunks.append(_mk(header + seg, repo=repo, path=rel,
                                  commit=commit, start_line=n.lineno,
                                  end_line=_end_line(n, n.lineno),
                                  kind="class_config",
                                  name=f"{node.name}.{name}",
                                  public_api=public))
 
            for m in node.body:
                if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                seg = ast.get_source_segment(src, m) or ""
                if m.name.startswith("__") and m.name.endswith("__") \
                        and len(seg) < 200:
                    continue
                _emit_callable(m, src, repo=repo, rel=rel, commit=commit,
                               chunks=chunks, parent=node.name,
                               parent_bases=bases, public_api=public)
 
    return chunks


# --------------------------------------------------------------------------
# prose chunking
# --------------------------------------------------------------------------

MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
# reStructuredText headings are a line of text followed by a line of punctuation.
RST_UNDERLINE = re.compile(r"^([=\-`:'\"~^_*+#<>])\1{2,}\s*$")


def _sections(lines: list[str], is_rst: bool) -> list[tuple[str, int, list[str]]]:
    """Return (heading, start_line, body_lines) tuples."""
    sections: list[tuple[str, int, list[str]]] = []
    heading, start, body = "", 1, []

    i = 0
    while i < len(lines):
        line = lines[i]
        new_heading, skip = None, 1

        if is_rst and i + 1 < len(lines) and RST_UNDERLINE.match(lines[i + 1]):
            if line.strip():
                new_heading, skip = line.strip(), 2
        else:
            m = MD_HEADING.match(line)
            if m:
                new_heading, skip = m.group(2).strip(), 1

        if new_heading is not None:
            if body:
                sections.append((heading, start, body))
            heading, start, body = new_heading, i + 1, []
            i += skip
            continue

        body.append(line)
        i += 1

    if body:
        sections.append((heading, start, body))
    return sections


def _is_autodoc_stub(text: str) -> bool:
    """Mostly Sphinx directives -> no information of its own."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    directive_like = sum(
        1 for l in lines
        if l.lstrip().startswith("..") or l.lstrip().startswith(":")
    )
    return AUTODOC_DIRECTIVE.search(text) is not None and \
        directive_like >= 0.5 * len(lines)


def chunk_prose(path: Path, repo: str, commit: str, root: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    title = Path(rel).stem.replace("_", " ").replace("-", " ")
    is_rst = path.suffix == ".rst"
 
    chunks: list[dict] = []
 
    # A short document is one idea. Splitting it on every heading turns, say, a
    # 500-byte pull-request template into four context-free fragments, each too
    # thin to retrieve well. If the whole file fits in a chunk, keep it whole.
    stripped = text.strip()
    if MIN_CHUNK_CHARS <= len(stripped) <= PROSE_MAX_CHARS \
            and not _is_autodoc_stub(stripped):
        header = f"File: {rel}\n{repo} docs > {title}\n\n"
        return [_mk(header + stripped, repo=repo, path=rel, commit=commit,
                    start_line=1, end_line=len(text.splitlines()),
                    kind="prose", name=title)]
 
    for heading, start_line, body in _sections(text.splitlines(), is_rst):
        body_text = "\n".join(body).strip()
        if len(body_text) < MIN_CHUNK_CHARS or _is_autodoc_stub(body_text):
            continue
 
        breadcrumb = f"{repo} docs > {title}"
        if heading:
            breadcrumb += f" > {heading}"
        header = f"File: {rel}\n{breadcrumb}\n\n"
 
        for i, piece in enumerate(
            _split_long(body_text, PROSE_MAX_CHARS, PROSE_OVERLAP_CHARS)
        ):
            suffix = "" if i == 0 else f" (part {i + 1})"
            chunks.append(_mk(
                header + piece,
                repo=repo, path=rel, commit=commit,
                start_line=start_line, end_line=start_line + len(body),
                kind="prose", name=(heading or title) + suffix,
            ))
    return chunks


# --------------------------------------------------------------------------
# notebook chunking
# --------------------------------------------------------------------------

def chunk_notebook(path: Path, repo: str, commit: str, root: Path) -> list[dict]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []

    rel = str(path.relative_to(root))
    title = Path(rel).stem.replace("_", " ").replace("-", " ")
    chunks: list[dict] = []
    buf: list[str] = []
    heading = ""

    def flush(idx: int, current_heading: str):
        text = "\n\n".join(buf).strip()
        if len(text) < MIN_CHUNK_CHARS:
            return
        header = f"File: {rel}\n{repo} tutorial > {title}"
        if current_heading:
            header += f" > {current_heading}"
        header += "\n\n"
        for i, piece in enumerate(
            _split_long(text, PROSE_MAX_CHARS, PROSE_OVERLAP_CHARS)
        ):
            suffix = "" if i == 0 else f" (part {i + 1})"
            chunks.append(_mk(
                header + piece,
                repo=repo, path=rel, commit=commit,
                start_line=idx, end_line=idx,
                kind="notebook",
                name=(current_heading or title) + suffix,
            ))

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            m = MD_HEADING.search(source)
            if m and buf:
                flush(idx, heading)
                buf.clear()
                heading = m.group(2).strip()
            elif m:
                heading = m.group(2).strip()
            buf.append(source)
        elif cell.get("cell_type") == "code":
            # Outputs are dropped: often megabytes of base64 images.
            buf.append(f"```python\n{source}\n```")

    flush(len(cells), heading)
    return chunks

# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

NEVER_DEDUPE = {"tasks.py", "noxfile.py", "Makefile"}

def dedupe_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Drop chunks whose embedded text duplicates one already kept.
 
    Repositories often hold the same file twice -- straxen keeps its tutorial
    notebooks in both `notebooks/` and `docs/source/`, and Sphinx copies files
    around at build time. Indexing both means retrieval returns the same
    passage two or three times and the prompt budget is spent on repeats.
 
    Comparison ignores the file path. Every chunk carries its own path in its
    header, so two copies of the same file differ by exactly that string and
    would otherwise hash differently -- the duplication we are trying to catch
    would slip through. Substituting a placeholder before hashing makes copies
    compare equal while keeping the path in the stored chunk, where citations
    need it.
 
    When two chunks tie, the one with the shallower path wins, which tends to
    pick the original rather than a copy nested under a docs tree. Ties on
    depth are broken alphabetically so the result is deterministic.
 
    Returns the kept chunks in their original order, plus how many were
    dropped.
    """
    def rank(c: dict) -> tuple[int, str]:
        return (len(Path(c["path"]).parts), c["path"])
 
    def fingerprint(c: dict) -> str:
        body = c["text"].replace(c["path"], "<path>")
        if Path(c["path"]).name in NEVER_DEDUPE:
            body = c["repo"] + body
        return hashlib.sha1(body.encode("utf-8")).hexdigest()
 
    winner: dict[str, int] = {}
    for i, c in enumerate(chunks):
        key = fingerprint(c)
        best = winner.get(key)
        if best is None or rank(c) < rank(chunks[best]):
            winner[key] = i
 
    keep = set(winner.values())
    return [c for i, c in enumerate(chunks) if i in keep], len(chunks) - len(keep)


def documented_modules(root: Path) -> set[str]:
    """Module names appearing in autodoc directives -> treated as public API."""
    pat = re.compile(r"^\.\.\s+auto(?:module|class)::\s+(\S+)", re.M)
    found: set[str] = set()
    for rst in root.rglob("*.rst"):
        try:
            found.update(pat.findall(rst.read_text(errors="ignore")))
        except OSError:
            continue
    return found


def chunk_repo(root: Path, repo: str, commit: str,
               ) -> list[dict]:
    """Chunk every supported file under `root`.
 
    `visibility` is stamped on every chunk. Use "internal" for anything from a
    private repository or hand-copied collaboration documentation, so a public
    build can filter it out. Getting this wrong is how private material ends
    up committed to a public repo alongside the index.
    """

    chunks: list[dict] = []
    # os.walk with followlinks=False, rather than rglob, because rglob
    # descends into symlinked directories. straxen symlinks
    # docs/source/tutorials -> notebooks/tutorials, and the files inside are
    # not themselves symlinks, so a per-file check cannot see it. Walking
    # without following links skips the whole tree once.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIR_PARTS
                             and not (here / d).is_symlink())
        for fname in sorted(filenames):
            path = here / fname
            if path.is_symlink():
                continue
            rel = path.relative_to(root)
            if should_skip(rel):
                continue
            if path.suffix == ".py":
                chunks.extend(chunk_python(path, repo, commit, root))
            elif path.suffix in {".md", ".rst"}:
                chunks.extend(chunk_prose(path, repo, commit, root))
            elif path.suffix == ".ipynb":
                chunks.extend(chunk_notebook(path, repo, commit, root))
 
    return chunks


def main():
    import argparse
    import subprocess
    from collections import Counter

    ap = argparse.ArgumentParser(description="Chunk a repo and report on it.")
    ap.add_argument("repo_path", type=Path)
    ap.add_argument("--name", help="repo name (defaults to directory name)")
    ap.add_argument("--out", type=Path, help="write chunks as jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="print N random chunks for inspection")
    ap.add_argument("--chars", type=int, default=700,
                    help="how much of each sampled chunk to print")
    ap.add_argument("--grep", help="only show chunks whose name or path matches")
    ap.add_argument("--kind", help="only show chunks of this kind")
    ap.add_argument("--context", action="store_true",
                    help="print context_text instead of text")
    args = ap.parse_args()

    name = args.name or args.repo_path.name
    commit = subprocess.run(
        ["git", "-C", str(args.repo_path), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    chunks = chunk_repo(args.repo_path, name, commit)

    kinds = Counter(c["kind"] for c in chunks)
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"repo:    {name}")
    print(f"commit:  {commit[:10]}")
    print(f"chunks:  {len(chunks)}")
    print(f"chars:   {sum(len(c['text']) for c in chunks)}")
    for kind, n in kinds.most_common():
        print(f"  {kind:20s} {n}")
    if lengths:
        print(f"embedded chars: median {lengths[len(lengths) // 2]}, "
              f"p95 {lengths[int(len(lengths) * 0.95)]}, max {lengths[-1]}")
    expanded = sum(1 for c in chunks if c["context_text"] != c["text"])
    print(f"chunks with expanded context: {expanded}")
    statuses = Counter(c["status"] for c in chunks if c["status"])
    if statuses:
        print("status:", dict(statuses))

    selected = chunks
    if args.grep:
        q = args.grep.lower()
        selected = [c for c in selected
                    if q in c["name"].lower() or q in c["path"].lower()]
    if args.kind:
        selected = [c for c in selected if c["kind"] == args.kind]

    if args.grep or args.kind:
        print(f"\nmatched {len(selected)} chunks")
        for c in selected:
            print(f"\n--- [{c['kind']}] {c['repo']}/{c['path']}:"
                  f"{c['start_line']} ({c['name']}) "
                  f"[{len(c['text'])} embedded / {len(c['context_text'])} context]\n")
            print((c["context_text"] if args.context else c["text"])[:args.chars])

    elif args.sample:
        import random
        print("\n" + "=" * 70)
        for c in random.sample(chunks, min(args.sample, len(chunks))):
            print(f"\n--- [{c['kind']}] {c['repo']}/{c['path']}:"
                  f"{c['start_line']} ({c['name']})\n")
            print((c["context_text"] if args.context else c["text"])[:args.chars])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"\nwrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
