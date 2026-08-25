"""Tests for xenonrag.chunking.
 
Everything here runs against small synthetic files written to tmp_path, so
the suite is fast and does not depend on the cloned repos being present.
 
The most valuable tests are the invariants at the bottom: they encode the
properties that must hold no matter how the chunking strategy changes, and
they are what will catch a regression six months from now.
"""
 
from __future__ import annotations
 
import ast
import json
import textwrap
from pathlib import Path
 
import pytest
 
from xenonrag import chunking as ch
 
 
# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
 
PLUGIN_SRC = textwrap.dedent('''
    """Module for merging S2 peaks into larger signals for analysis."""
    import strax
    import straxen
 
 
    class MergedS2s(strax.OverlapWindowPlugin):
        """Merge together peaklets if peak finding favours that.
 
        This is the long explanation of what merging does and why it
        matters for the analysis chain.
        """
 
        __version__ = "1.2.0"
        depends_on = ("peaklets", "peaklet_classification")
        provides = "merged_s2s"
        data_kind = "merged_s2s"
 
        merge_without_s1 = strax.Option(
            "merge_without_s1",
            default=True,
            help="If true, S1s will be igored during the merging. "
                 "It is recommended for the peaklet classification.",
        )
        gain_model = straxen.URLConfig(
            infer_type=False,
            help="PMT gain model. Specify as (str(model_config), str(version), "
                 "nT = True)",
        )
 
        def infer_dtype(self):
            return strax.merged_dtype()
 
        def compute(self, peaklets, lone_hits):
            """Merge the peaklets."""
            merged = strax.merge_peaks(peaklets)
            return merged
 
        def __repr__(self):
            return "MergedS2s"
 
 
    def helper(x, y):
        """Add two numbers."""
        return x + y
''').strip()
 
 
STATUS_SRC = textwrap.dedent('''
    class Plugin:
        """Base class users subclass."""
 
        def compute(self, chunk):
            """Compute the data. Subclasses must override this."""
            raise NotImplementedError
 
        def infer_dtype(self):
            raise NotImplementedError
 
 
    class ZipDirectory(Plugin):
        """Read-only zip storage."""
 
        def remove(self, key):
            """Zip archives cannot be modified in place."""
            self.log.warning("cannot remove from a zip")
            cleanup(self.path)
            raise NotImplementedError("zip storage is read only")
 
 
    class OldWidget:
        def __init__(self, name):
            """Interactive selection plot."""
            raise NotImplementedError(
                "This function does not work with the latest bokeh version."
                " Please contact tech-support; otherwise we remove it."
            )
 
 
    def mentions_but_does_not_raise(x):
        """A normal function.
 
        Note: callers used to get NotImplementedError here, but that was
        fixed and the exception is no longer raised.
        """
        total = 0
        for i in range(x):
            total += i
        return total
 
 
    def ordinary(a, b):
        """Add two numbers together."""
        return a + b
''').strip()
 

# A function whose signature alone would fill the embedding window: many
# parameters with long path defaults, exactly like straxen's context builders.
WIDE_SIG_SRC = textwrap.dedent('''
    def build_context(
        output_folder="./strax_data",
        raw_paths=["/dali/lgrandi/xenonnt/raw", "/dali/lgrandi/xenonnt/raw_2"],
        processed_paths=[
            "/project/lgrandi/xenonnt/processed_sr2_offline_round_1",
            "/project/lgrandi/xenonnt/processed_sr2_offline_round_2",
            "/project/lgrandi/xenonnt/processed_sr2_offline_round_3",
            "/project2/lgrandi/xenonnt/processed_sr2_offline_round_4",
        ],
        include_rucio_remote=False,
        include_online_monitor=False,
        minimum_run_number=7157,
        **kwargs,
    ):
        """Build an analysis context with corrections configuration applied.
 
        Use the versioned variants if you need a specific corrections version.
        """
        step_0 = configure_storage_frontend(0, output_folder, raw_paths)
        step_1 = configure_storage_frontend(1, output_folder, raw_paths)
        step_2 = configure_storage_frontend(2, output_folder, raw_paths)
        step_3 = configure_storage_frontend(3, output_folder, raw_paths)
        step_4 = configure_storage_frontend(4, output_folder, raw_paths)
        step_5 = configure_storage_frontend(5, output_folder, raw_paths)
        step_6 = configure_storage_frontend(6, output_folder, raw_paths)
        step_7 = configure_storage_frontend(7, output_folder, raw_paths)
        step_8 = configure_storage_frontend(8, output_folder, raw_paths)
        step_9 = configure_storage_frontend(9, output_folder, raw_paths)
        step_10 = configure_storage_frontend(10, output_folder, raw_paths)
        step_11 = configure_storage_frontend(11, output_folder, raw_paths)
        step_12 = configure_storage_frontend(12, output_folder, raw_paths)
        step_13 = configure_storage_frontend(13, output_folder, raw_paths)
        step_14 = configure_storage_frontend(14, output_folder, raw_paths)
        step_15 = configure_storage_frontend(15, output_folder, raw_paths)
        step_16 = configure_storage_frontend(16, output_folder, raw_paths)
        step_17 = configure_storage_frontend(17, output_folder, raw_paths)
        step_18 = configure_storage_frontend(18, output_folder, raw_paths)
        step_19 = configure_storage_frontend(19, output_folder, raw_paths)
        step_20 = configure_storage_frontend(20, output_folder, raw_paths)
        step_21 = configure_storage_frontend(21, output_folder, raw_paths)
        step_22 = configure_storage_frontend(22, output_folder, raw_paths)
        step_23 = configure_storage_frontend(23, output_folder, raw_paths)
        step_24 = configure_storage_frontend(24, output_folder, raw_paths)
        step_25 = configure_storage_frontend(25, output_folder, raw_paths)
        step_26 = configure_storage_frontend(26, output_folder, raw_paths)
        step_27 = configure_storage_frontend(27, output_folder, raw_paths)
        step_28 = configure_storage_frontend(28, output_folder, raw_paths)
        step_29 = configure_storage_frontend(29, output_folder, raw_paths)
        return Context(**kwargs)
''').strip()

 
LONG_FUNC_SRC = textwrap.dedent('''
    def enormous(a, b, c=1):
        """Do a great many things.
 
        Second paragraph of the docstring.
        """
    ''') + "\n".join(f"    value_{i} = compute_something({i}, a, b, c)"
                     for i in range(200)) + "\n    return value_0\n"
 
 
NO_BLANK_LINES_SRC = "def dense(x):\n" + "\n".join(
    f"    x = x + {i}" for i in range(150)
) + "\n    return x\n"
 
 
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "fakerepo"
    (root / "pkg" / "plugins").mkdir(parents=True)
    (root / "docs" / "source" / "reference").mkdir(parents=True)
    (root / "tests").mkdir()
 
    (root / "pkg" / "plugins" / "merged_s2s.py").write_text(PLUGIN_SRC)
    (root / "pkg" / "long.py").write_text(LONG_FUNC_SRC)
    (root / "pkg" / "dense.py").write_text(NO_BLANK_LINES_SRC)
    (root / "pkg" / "status.py").write_text(STATUS_SRC)
    (root / "pkg" / "wide.py").write_text(WIDE_SIG_SRC)
    (root / "tests" / "test_thing.py").write_text("def test_x():\n    assert 1\n")
 
    (root / "HISTORY.md").write_text(
        "# Changelog\n\n## 2.2.5\n\n" + ("* fixed a thing by @someone\n" * 200)
    )
    (root / "docs" / "source" / "reference" / "pkg.rst").write_text(
        "Module contents\n---------------\n\n"
        ".. automodule:: pkg.plugins.merged_s2s\n"
        "   :members:\n"
        "   :undoc-members:\n"
        "   :show-inheritance:\n"
    )
    (root / "docs" / "source" / "guide.md").write_text(
        "# Getting started\n\n"
        + "This paragraph explains how to load data for a run in the framework. " * 4
        + "\n\n## Loading data\n\n"
        + "Call the context and ask it for the data type you want. " * 6
    )
 
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": ["# Tutorial\n", "\n", "How to load data from a run.\n"]},
        {"cell_type": "code",
         "source": ["import straxen\n", "st = straxen.contexts.xenonnt_online()\n"],
         "outputs": [{"data": {"image/png": "A" * 50000}}]},
        {"cell_type": "markdown",
         "source": ["## Plotting\n", "\n",
                    "Now we plot the peaks we just loaded from disk.\n"]},
        {"cell_type": "code", "source": ["st.plot_peaks(run_id)\n"]},
    ]}
    (root / "docs" / "tutorial.ipynb").write_text(json.dumps(nb))
    return root
 
 
def chunks_of(repo: Path) -> list[dict]:
    return ch.chunk_repo(repo, "fakerepo", "deadbeef")
 
 
def by_kind(chunks: list[dict], kind: str) -> list[dict]:
    return [c for c in chunks if c["kind"] == kind]
 
 
def named(chunks: list[dict], name: str) -> dict:
    matches = [c for c in chunks if c["name"] == name]
    assert matches, f"no chunk named {name!r}; have {[c['name'] for c in chunks]}"
    return matches[0]
 
 
# --------------------------------------------------------------------------
# helpers: _split_long, _tail_lines, _truncate_lines
# --------------------------------------------------------------------------
 
def test_split_long_leaves_short_text_alone():
    assert ch._split_long("short", 100, 10) == ["short"]
 
 
def test_split_long_respects_max_chars():
    text = "\n\n".join("word " * 50 for _ in range(20))
    for piece in ch._split_long(text, 500, 50):
        assert len(piece) <= 500
 
 
def test_split_long_handles_text_with_no_blank_lines():
    """Regression: dense code has no paragraph breaks to split on."""
    text = "\n".join(f"line {i}" for i in range(500))
    pieces = ch._split_long(text, 400, 40)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 400
 
 
def test_split_long_terminates_on_one_enormous_line():
    """A single line longer than max_chars must not loop forever."""
    pieces = ch._split_long("x" * 5000, 400, 40)
    assert len(pieces) > 1
    assert all(len(p) <= 400 for p in pieces)
 
 
def test_overlap_starts_at_a_line_boundary():
    """Regression: continuation chunks used to start mid-token."""
    text = "\n\n".join(f"paragraph_{i} " + "filler " * 40 for i in range(15))
    pieces = ch._split_long(text, 500, 100)
    assert len(pieces) > 1
    original_lines = set(text.splitlines())
    for piece in pieces[1:]:
        first = piece.splitlines()[0]
        assert first in original_lines, f"mid-token start: {first[:60]!r}"
 

def test_tail_lines_never_exceeds_its_budget():
    """Regression: a single line longer than the overlap budget used to be
    returned whole, prepending a full paragraph to the next chunk and
    duplicating it across two chunks."""
    one_long_line = "x" * 900
    assert ch._tail_lines(one_long_line, 150) == ""
 
    text = "short\n" + "y" * 900
    assert len(ch._tail_lines(text, 150)) <= 150
 
 
def test_split_long_does_not_duplicate_a_long_paragraph():
    """Two pieces must never begin with the same text."""
    p1 = "A " * 450          # one 900-char line, no internal breaks
    p2 = "\n".join(f"* bullet {i} " + "b" * 150 for i in range(4))
    p3 = "C " * 230
    pieces = ch._split_long(f"{p1}\n\n{p2}\n\n{p3}", 1200, 150)
    heads = [p.strip()[:120] for p in pieces]
    assert len(heads) == len(set(heads)), "a piece was duplicated"


def test_tail_lines_returns_whole_lines():
    text = "alpha\nbeta\ngamma\ndelta"
    tail = ch._tail_lines(text, 12)
    assert tail in ("gamma\ndelta", "delta")
    assert not tail.startswith("amma")
 
 
def test_truncate_lines_returns_whole_lines():
    text = "alpha\nbeta\ngamma"
    assert ch._truncate_lines(text, 10) == "alpha\nbeta"
    assert ch._truncate_lines(text, 1000) == text
 
 
# --------------------------------------------------------------------------
# config detection
# --------------------------------------------------------------------------
 
@pytest.mark.parametrize("src,expected", [
    ('x = strax.Option("a", default=1)', True),
    ('x = straxen.URLConfig(help="h")', True),
    ('x = Option(1)', True),
    ('__version__ = "1.0.0"', False),
    ('depends_on = ("peaklets",)', False),
    ('x = some_function(1)', False),
    ('x: int = 5', False),
])
def test_is_config(src, expected):
    node = ast.parse(src).body[0]
    assert ch._is_config(node) is expected
 
 
def test_target_name_handles_annassign():
    assert ch._target_name(ast.parse("x: int = 5").body[0]) == "x"
    assert ch._target_name(ast.parse("y = 5").body[0]) == "y"
 
 
# --------------------------------------------------------------------------
# class cards
# --------------------------------------------------------------------------
 
def test_class_card_contains_identity(repo):
    card = named(chunks_of(repo), "MergedS2s")
    text = card["context_text"]
    assert "MergedS2s" in text
    assert "strax.OverlapWindowPlugin" in text
    assert "Merge together peaklets" in text
    assert "depends_on" in text
    assert "provides" in text
 
 
def test_class_card_lists_config_names_not_values(repo):
    """The card names the options; the full declarations live elsewhere."""
    card = named(chunks_of(repo), "MergedS2s")
    text = card["context_text"]
    assert "merge_without_s1" in text
    assert "gain_model" in text
    assert "Configuration options (2)" in text
    assert "It is recommended for the peaklet classification" not in text
 
 
def test_class_card_excludes_method_bodies(repo):
    """Regression: ast.walk used to duplicate every method inside the card."""
    card = named(chunks_of(repo), "MergedS2s")
    text = card["context_text"]
    assert "def compute" in text          # signature is listed
    assert "strax.merge_peaks" not in text  # body is not
 
 
def test_config_chunk_holds_full_declarations(repo):
    configs = by_kind(chunks_of(repo), "class_config")
    assert len(configs) >= 1
    joined = " ".join(c["context_text"] for c in configs)
    assert "It is recommended for the peaklet classification" in joined
    assert "PMT gain model" in joined
 
 
def test_class_with_no_configs_emits_no_config_chunk(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "m.py").write_text("class Plain:\n    x = 1\n\n    def go(self):\n        return 1\n")
    assert by_kind(chunks_of(root), "class_config") == []
 
 
# --------------------------------------------------------------------------
# methods and functions
# --------------------------------------------------------------------------
 
def test_methods_become_their_own_chunks(repo):
    method = named(chunks_of(repo), "MergedS2s.compute")
    assert method["kind"] == "method"
    assert "strax.merge_peaks" in method["context_text"]
    assert "Class: MergedS2s" in method["text"]
 
 
def test_trivial_dunder_methods_are_skipped(repo):
    names = [c["name"] for c in chunks_of(repo)]
    assert "MergedS2s.__repr__" not in names
 
 
def test_top_level_function_is_chunked(repo):
    fn = named(chunks_of(repo), "helper")
    assert fn["kind"] == "function"
    assert "Add two numbers" in fn["context_text"]
 
 
def test_very_long_function_becomes_a_summary(repo):
    summaries = by_kind(chunks_of(repo), "function_summary")
    names = [c["name"] for c in summaries]
    assert "enormous" in names, f"long function not summarised; got {names}"
    s = next(c for c in summaries if c["name"] == "enormous")
    assert "Do a great many things" in s["text"]
    assert "def enormous(a, b, c=1)" in s["text"]
    # small-to-big: embedded form is short, prompt context is the whole thing
    assert len(s["text"]) < len(s["context_text"])
    assert "value_199" in s["context_text"]
 
 
def test_huge_signature_is_capped(repo):
    """Regression: a 1200-char signature consumed 83% of the embedding budget
    and pushed the docstring out entirely."""
    c = named(chunks_of(repo), "build_context")
    assert "..." in c["text"], "signature was not truncated"
    assert "round_4" not in c["text"], "long defaults still in the embedded text"
 
 
def test_docstring_survives_a_huge_signature(repo):
    """The description is what makes a chunk findable; it must not be the
    thing that gets truncated."""
    c = named(chunks_of(repo), "build_context")
    assert "corrections configuration" in c["text"]
    assert c["text"].index("corrections configuration") < c["text"].index("def build_context(")
 
 
def test_wide_signature_routes_to_summary(repo):
    """Summarised for signature width, not only for length."""
    c = named(chunks_of(repo), "build_context")
    assert c["kind"] == "function_summary"
 
 
def test_full_source_is_still_available_to_the_llm(repo):
    c = named(chunks_of(repo), "build_context")
    assert "round_4" in c["context_text"]
    assert len(c["context_text"]) > len(c["text"])
 
 
def test_module_docstring_chunk(repo):
    mods = by_kind(chunks_of(repo), "module_docstring")
    assert any("merging S2 peaks" in c["text"] for c in mods)
 
 
# --------------------------------------------------------------------------
# prose, notebooks, filtering
# --------------------------------------------------------------------------
 
def test_changelog_is_skipped(repo):
    assert all("HISTORY" not in c["path"] for c in chunks_of(repo))
 
 
def test_autodoc_stub_is_skipped(repo):
    assert all(".. automodule::" not in c["text"] for c in chunks_of(repo))


def test_symlinked_directory_is_not_walked_twice(tmp_path):
    """straxen symlinks docs/source/tutorials -> notebooks/tutorials. The
    files inside are not themselves symlinks, so a per-file check misses it."""
    import os, json as _json
    root = tmp_path / "symrepo"
    (root / "notebooks" / "tutorials").mkdir(parents=True)
    (root / "docs" / "source").mkdir(parents=True)
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": ["# Loading data\n", "\n",
                    "How to load data from a run in the framework.\n"]},
        {"cell_type": "code",
         "source": ["st = straxen.contexts.xenonnt_online()\n",
                    "peaks = st.get_array(run_id, 'peaks')\n"]},
    ]}
    (root / "notebooks" / "tutorials" / "demo.ipynb").write_text(_json.dumps(nb))
    os.symlink("../../notebooks/tutorials", root / "docs" / "source" / "tutorials")
 
    chunks = chunks_of(root)
    assert chunks
    assert all("docs/source/tutorials" not in c["path"] for c in chunks)
    assert ch.dedupe_chunks(chunks)[1] == 0, "symlink slipped through to dedupe"
 
 
def test_build_tooling_is_skipped(tmp_path):
    """tasks.py and friends are copied verbatim between projects and answer
    no question a user of the analysis framework would ask."""
    root = tmp_path / "toolrepo"
    root.mkdir()
    (root / "tasks.py").write_text(
        'def lint_flake8(c):\n    """Run flake8."""\n    c.run("flake8")\n')
    (root / "noxfile.py").write_text('def tests(session):\n    """Run tests."""\n    pass\n')
    (root / "real.py").write_text('def merge(a, b):\n    """Merge peaks."""\n    return a + b\n')
    names = [c["name"] for c in chunks_of(root)]
    assert "merge" in names
    assert "lint_flake8" not in names
    assert "tests" not in names
 
 
def test_tests_directory_is_skipped(repo):
    assert all("tests/" not in c["path"] for c in chunks_of(repo))
 
 
def test_prose_keeps_real_documentation(repo):
    prose = by_kind(chunks_of(repo), "prose")
    assert prose
    assert any("Getting started" in c["name"] for c in prose)
    assert all("docs" in c["path"] for c in prose)
 
 
def test_prose_chunks_carry_a_breadcrumb(repo):
    for c in by_kind(chunks_of(repo), "prose"):
        assert "fakerepo docs >" in c["text"]
 
 
def test_notebook_pairs_markdown_with_following_code(repo):
    nbs = by_kind(chunks_of(repo), "notebook")
    assert nbs
    joined = " ".join(c["text"] for c in nbs)
    assert "How to load data" in joined
    assert "xenonnt_online()" in joined
 
 
def test_notebook_outputs_are_dropped(repo):
    """Regression: cell outputs can be megabytes of base64 image data."""
    for c in by_kind(chunks_of(repo), "notebook"):
        assert "A" * 200 not in c["text"]
        assert "A" * 200 not in c["context_text"]
 
 
def test_notebook_splits_on_headings(repo):
    names = [c["name"] for c in by_kind(chunks_of(repo), "notebook")]
    assert any("Tutorial" in n for n in names)
    assert any("Plotting" in n for n in names)
 
 
def test_public_api_flag_from_autodoc_directives(repo):
    """The autodoc stubs are skipped as content but used as metadata."""
    card = named(chunks_of(repo), "MergedS2s")
    assert card["public_api"] is True
    # nothing marks pkg/long.py as public
    assert named(chunks_of(repo), "enormous")["public_api"] is False
 
 
# --------------------------------------------------------------------------
# deprecation / abstract classification
# --------------------------------------------------------------------------
 
def test_abstract_method_is_flagged(repo):
    """A stub whose whole body is `raise NotImplementedError`."""
    assert named(chunks_of(repo), "Plugin.compute")["status"] == "abstract"
    assert named(chunks_of(repo), "Plugin.infer_dtype")["status"] == "abstract"
 
 
def test_method_that_refuses_on_purpose_is_unsupported(repo):
    """Real logic before the raise -> a deliberate not-supported case,
    not an interface stub."""
    assert named(chunks_of(repo), "ZipDirectory.remove")["status"] == "unsupported"
 
 
def test_deprecated_function_is_flagged(repo):
    assert named(chunks_of(repo), "OldWidget.__init__")["status"] == "deprecated"
 
 
def test_mentioning_the_exception_is_not_enough(repo):
    """Regression: a substring search would misclassify this."""
    c = named(chunks_of(repo), "mentions_but_does_not_raise")
    assert c["status"] is None
 
 
def test_ordinary_function_has_no_status(repo):
    assert named(chunks_of(repo), "ordinary")["status"] is None
 
 
def test_deprecation_beats_abstract(repo):
    """OldWidget.__init__ raises NotImplementedError *and* says it is going
    away. The deprecation reading is the useful one."""
    assert named(chunks_of(repo), "OldWidget.__init__")["status"] != "abstract"
 
 
def test_all_parts_of_a_split_function_share_a_status(repo):
    """Status is a property of the function, not of the fragment."""
    for c in chunks_of(repo):
        siblings = [o for o in chunks_of(repo)
                    if o["path"] == c["path"] and o["start_line"] == c["start_line"]
                    and o["kind"] == c["kind"]]
        assert len({o["status"] for o in siblings}) == 1
 
 
def test_raises_not_implemented_is_structural():
    src = "def f():\n    # raise NotImplementedError one day\n    return 1\n"
    node = ast.parse(src).body[0]
    assert ch._raises_not_implemented(node) is False
 
    src = "def f():\n    raise NotImplementedError()\n"
    node = ast.parse(src).body[0]
    assert ch._raises_not_implemented(node) is True
 

 # --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------

def _c(path, body="same body text here for the chunk"):
    """Build a chunk the way the real chunkers do: path in the header."""
    text = f"File: {path}\n\n{body}"
    return {"text": text, "context_text": text, "repo": "r", "path": path,
            "commit": "c", "start_line": 1, "end_line": 2, "kind": "prose",
            "name": "n", "public_api": False, "status": None}


def test_dedupe_removes_identical_text():
    kept, dropped = ch.dedupe_chunks([_c("a/x.md"), _c("b/y.md")])
    assert dropped == 1
    assert len(kept) == 1


def test_dedupe_keeps_the_shallower_path():
    """straxen keeps notebooks in both notebooks/ and docs/source/;
    the shallower path is the original."""
    kept, _ = ch.dedupe_chunks([
        _c("docs/source/tutorials/demo.ipynb"),
        _c("notebooks/tutorials/demo.ipynb"),
    ])
    assert kept[0]["path"] == "notebooks/tutorials/demo.ipynb"


def test_dedupe_ignores_the_path_when_comparing():
    """Regression: the path lives inside the chunk text, so two copies of the
    same file hash differently unless it is normalised out first."""
    a = _c("notebooks/tutorials/demo.ipynb")
    b = _c("docs/source/tutorials/demo.ipynb")
    assert a["text"] != b["text"]          # genuinely different strings
    kept, dropped = ch.dedupe_chunks([a, b])
    assert dropped == 1, "path difference defeated the duplicate check"
    assert kept[0]["path"] == "notebooks/tutorials/demo.ipynb"


def test_dedupe_keeps_the_path_in_the_surviving_chunk():
    """Normalisation is only for comparison; citations still need the path."""
    kept, _ = ch.dedupe_chunks([_c("a/x.md"), _c("b/x.md")])
    assert "<path>" not in kept[0]["text"]
    assert kept[0]["path"] in kept[0]["text"]


def test_dedupe_leaves_distinct_chunks_alone():
    kept, dropped = ch.dedupe_chunks([_c("a.md", "first body"),
                                      _c("b.md", "second body")])
    assert dropped == 0
    assert len(kept) == 2


def test_dedupe_preserves_order():
    chunks = [_c("a.md", "one"), _c("b.md", "two"), _c("c.md", "three")]
    kept, _ = ch.dedupe_chunks(chunks)
    assert [c["path"] for c in kept] == ["a.md", "b.md", "c.md"]


def test_dedupe_is_deterministic():
    chunks = [_c("x/a.md"), _c("y/a.md"), _c("z/a.md")]
    assert ch.dedupe_chunks(chunks)[0] == ch.dedupe_chunks(chunks)[0]


def test_dedupe_handles_empty_input():
    assert ch.dedupe_chunks([]) == ([], 0)
 
# --------------------------------------------------------------------------
# invariants -- these are the regression guards
# --------------------------------------------------------------------------
 
def test_every_embedded_text_fits_the_embedding_window(repo):
    """The embedder silently truncates past its limit, so nothing may exceed it."""
    for c in chunks_of(repo):
        assert len(c["text"]) <= ch.EMBED_MAX_CHARS, \
            f"{c['kind']} {c['name']} is {len(c['text'])} chars"
 
 
def test_no_chunk_is_empty(repo):
    for c in chunks_of(repo):
        assert c["text"].strip()
        assert c["context_text"].strip()
 
 
def test_context_text_is_never_shorter_than_text(repo):
    for c in chunks_of(repo):
        assert len(c["context_text"]) >= len(c["text"])
 
 
def test_every_chunk_has_required_metadata(repo):
    required = {"text", "context_text", "repo", "path", "commit",
                "start_line", "end_line", "kind", "name", "public_api",
                "status"}
    for c in chunks_of(repo):
        assert required <= set(c)
        assert c["start_line"] >= 1
        assert c["end_line"] >= c["start_line"]
        assert not Path(c["path"]).is_absolute()
        assert c["status"] in (None, "abstract", "unsupported", "deprecated")
 
 
def test_every_chunk_is_json_serialisable(repo):
    for c in chunks_of(repo):
        json.loads(json.dumps(c))
 
 
def test_chunking_is_deterministic(repo):
    assert chunks_of(repo) == chunks_of(repo)
 
 
def test_syntax_error_file_is_skipped_not_fatal(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "broken.py").write_text("def oops(:\n    pass\n")
    (root / "fine.py").write_text('def ok():\n    """Docs."""\n    return 1\n')
    names = [c["name"] for c in chunks_of(root)]
    assert "ok" in names
 
 
def test_empty_repo_returns_nothing(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert chunks_of(root) == []