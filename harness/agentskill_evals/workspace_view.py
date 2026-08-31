"""Shared, agent-agnostic views over a run's workspace — the file tree and
inlined file contents — used by BOTH the judge (to grade) and the per-cell
report (to show the human everything the model produced).

The judge takes a deliberately small slice (a few files, truncated) to keep its
prompt cheap; the report shows everything the model produced, in full. Both ride
the same walk so they never diverge on what counts as "the model's output"
(e.g. both exclude the provisioned skill dirs and VCS/build noise).
"""

from __future__ import annotations

import os
import shlex
from typing import Any
from collections.abc import Iterable, Iterator

# Budgets for the judge's compact view (the report passes None == no file-count cap).
JUDGE_MAX_FILES = 60
# The judge grades with tools disabled, so what is inlined here is the ONLY evidence it
# has. A cap that drops a produced file makes the judge fail rubric items it cannot see —
# so keep the file cap above what a realistic multi-project workspace produces, and let a
# long file truncate (below) rather than vanish. Both losses are now announced in-band.
JUDGE_MAX_INLINE_FILES = 20
JUDGE_MAX_INLINE_BYTES = 4000
# The report inlines every text file, but per-file only up to this many bytes (with a
# truncation note) — a run that legitimately produces a multi-MB CSV/JSON export must not
# balloon report.md; the full file is still in workspace/.
REPORT_MAX_INLINE_BYTES = 200_000

# Source/config extensions whose CONTENTS get inlined. An extension missing here is
# treated as binary and silently reduced to a filename in the tree — which, for the
# toolless judge, means grading a file it never saw. Keep it wide: the byte/file budgets
# above bound the cost, not this list.
_TEXT_EXT = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".cfg",
             ".ini", ".js", ".ts", ".html", ".css", ".sh", ".csv",
             # .NET / MSBuild
             ".cs", ".fs", ".vb", ".razor", ".cshtml", ".xaml",
             ".csproj", ".fsproj", ".vbproj", ".sln", ".slnx",
             ".props", ".targets", ".config", ".nuspec",
             # other common source/config the walk can meet
             ".jsx", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".rb", ".java", ".kt",
             ".c", ".h", ".cc", ".cpp", ".hpp", ".php", ".pl", ".swift", ".scala",
             ".sql", ".xml", ".svg", ".bash", ".zsh", ".ps1", ".psm1", ".psd1",
             ".dockerfile", ".tf", ".tfvars", ".proto",
             ".graphql", ".rst", ".tex", ".lua", ".r", ".jl", ".dart", ".ex", ".exs"}

# Files with NO extension to match on. `os.path.splitext(".editorconfig")` yields
# (".editorconfig", "") — a leading dot is not an extension — so a dotfile can never be
# recognized by _TEXT_EXT no matter what is put in it. Matched case-insensitively on the
# basename instead. Without this an `.editorconfig`-focused eval grades against nothing.
_TEXT_NAMES = {".editorconfig", ".gitignore", ".gitattributes", ".dockerignore",
               ".env", ".npmrc", ".nvmrc", ".prettierrc", ".eslintrc", ".babelrc",
               "dockerfile", "makefile", "readme", "license", "notice", "codeowners"}


def _is_text(path: str) -> bool:
    """True if this file's CONTENTS should be inlined — by extension, or, for a file that has
    none (dotfiles above all), by its basename."""
    base = os.path.basename(path).lower()
    if base in _TEXT_NAMES:
        return True
    return os.path.splitext(base)[1] in _TEXT_EXT


# Provisioned skills are inputs, not model output; .git/node_modules/etc. are noise.
_SKILL_DIRS = (".claude", ".agents", ".antigravity", ".codex")
_SKIP_DIRS = {".git", "node_modules", "__pycache__"}


def _is_skill_dir(rel_root: str) -> bool:
    """True if `rel_root` IS one of the provisioned skill dirs, or a path underneath one — a path
    segment match, not a bare string prefix (a real dir named e.g. `.codexnotes` must NOT match
    `.codex`, or the model's actual output would silently vanish from the report)."""
    return any(rel_root == d or rel_root.startswith(d + os.sep) for d in _SKILL_DIRS)


def _iter_files(workdir: str) -> Iterator[tuple[str, str]]:
    """Yield (abspath, relpath) for files the model could have produced —
    excluding VCS/build noise and the provisioned skill dirs."""
    for root, dirs, files in os.walk(workdir):
        dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_DIRS]
        rel_root = os.path.relpath(root, workdir)
        if _is_skill_dir(rel_root):
            continue
        for f in sorted(files):
            path = os.path.join(root, f)
            yield path, os.path.relpath(path, workdir)


def _is_under(base: str, path: str) -> bool:
    """True if the absolute `path` resolves inside the absolute `base` directory."""
    try:
        return os.path.commonpath([base, path]) == base
    except ValueError:        # different drives, etc. → treat as outside
        return False


def seeded_relpaths(spec: Any) -> set[str]:
    """Workspace-relative paths that were seeded into the workspace BEFORE the run (the
    fixture tree plus each `files:` destination) — inputs, not model output. Both the judge
    and the report annotate these so a seeded file is never credited as work the model did."""
    seeded: set[str] = set()
    if spec is None:
        return seeded
    try:
        fixture = spec.resolved_fixture()
    except Exception:
        fixture = None
    if fixture and os.path.isdir(fixture):
        for root, _dirs, files in os.walk(fixture):
            for f in files:
                seeded.add(os.path.relpath(os.path.join(root, f), fixture))
    try:
        for _src, dest in spec.resolved_files():
            seeded.add(dest)
    except Exception:
        pass
    return seeded


def resolve_trace_path(path: str, workdir: str) -> str:
    """Resolve a tool-trace path to an absolute path. The agent ran with cwd == the workspace, so a
    RELATIVE trace path is relative to the WORKSPACE — never the harness process cwd (resolving it
    against the process cwd is how an agent's relative ``README.md`` was wrongly matched to the
    repo's ``harness/README.md``). An ABSOLUTE trace path is kept as-is, so an agent that wrote to a
    mangled absolute path outside the workspace is still detected."""
    if os.path.isabs(path):
        return path
    return os.path.join(workdir, path)


def writes_outside_workspace(result: Any, workdir: str) -> list[str]:
    """Absolute paths the run created that landed OUTSIDE the workspace (e.g. the model wrote to an
    absolute path with a mangled run-id). Surfacing them lets the judge grade — and the report show —
    the artifact the run actually produced, not just whatever happened to land in the workspace.

    Uses ``realpath`` (not ``abspath``) on both sides of the containment check: a symlink inside the
    workspace pointing outside it (e.g. `workspace/link -> /elsewhere`) must resolve to its real
    target, or a write through that link would be wrongly counted as "inside"."""
    wd = os.path.realpath(workdir)
    out: list[str] = []
    seen: set[str] = set()
    for p in result.file_paths_touched():
        if not p:
            continue
        ap = os.path.realpath(resolve_trace_path(p, workdir))
        if ap in seen:
            continue
        if not _is_under(wd, ap) and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def leaked_skill_reads(
    result: Any, workdir: str, repo_root: str,
    repo_skill_names: Iterable[str], declared_names: Iterable[str],
) -> list[str]:
    """Absolute paths (or referenced script paths) this run touched that reach an UNDECLARED
    skill through the real, on-disk repo checkout rather than the provisioned workspace copy.

    Even with the eval workspace relocated to a tempdir outside this repo's working tree
    (``runner.py``), a run could still reach an undeclared skill some other way (e.g. searching
    the real disk by name, or a symlink planted inside the workspace pointing at the repo) — this
    is the residual safety net that catches it from the trace, so a leak is never silently
    reported as ``isolated: true``.

    Uses ``realpath`` (not ``abspath``): an agent that symlinks a workspace-local name at an
    undeclared skill (e.g. ``ln -s <repo>/sliderule-pipeline-direct-request evil`` then reads ``evil/SKILL.md``) would
    otherwise look textually "inside the workspace" and never get flagged at all.
    """
    leaked_names = set(repo_skill_names) - set(declared_names)
    if not leaked_names:
        return []
    root = os.path.realpath(repo_root)
    wd = os.path.realpath(workdir)
    hits: list[str] = []
    seen: set[str] = set()

    def _flag(candidate: str) -> None:
        if not candidate or candidate in seen:
            return
        ap = os.path.realpath(candidate)
        if not _is_under(root, ap) or _is_under(wd, ap):
            return
        rel_parts = os.path.relpath(ap, root).split(os.sep)
        if rel_parts and rel_parts[0] in leaked_names:
            seen.add(candidate)
            hits.append(ap)

    for p in result.file_paths_touched():
        _flag(resolve_trace_path(p, workdir))

    # Markers match literal text in a raw shell command string, so they must use the SAME
    # (unresolved) form the agent would actually have typed — not the realpath'd `root` above,
    # which e.g. on macOS resolves /var -> /private/var and would never textually match a
    # command referencing the ordinary /var path. `_flag` still realpath-resolves the matched
    # token for the actual containment decision.
    raw_root = os.path.abspath(repo_root)
    markers = [os.path.join(raw_root, name) for name in leaked_names]
    for cmd in result.commands():
        try:
            tokens = shlex.split(cmd)
        except ValueError:        # unbalanced quotes, etc. — fall back to whitespace split
            tokens = cmd.split()
        for tok in tokens:
            for marker in markers:
                # `in`, not `startswith`: the marker can be embedded mid-token (e.g.
                # --script=/repo/sliderule-pipeline-direct-request/scripts/tool.py or --path=/repo/... ), not just
                # at the start. Flag from where the marker begins, not the whole raw token (a
                # leading flag like "--script=" isn't a path itself, so realpath-ing it whole
                # would resolve it relative to cwd instead of recognizing the embedded absolute
                # path).
                idx = tok.find(marker)
                if idx != -1:
                    _flag(tok[idx:])

    return hits


def file_tree(workdir: str, extra: list[str] = (), max_files: int | None = None,
              seeded: Iterable[str] = ()) -> str:
    """A flat listing of every file under `workdir` (skill dirs / noise excluded), plus any
    `extra` paths written outside it. `max_files=None` lists everything (the report); the judge
    passes a cap. Paths in `seeded` (workspace-relative) are annotated as pre-seeded inputs."""
    seeded_set = set(seeded or ())
    lines: list[str] = []
    count = 0
    truncated = False
    for _abs, rel in _iter_files(workdir):
        if max_files is not None and count >= max_files:
            truncated = True
            break
        tag = "   [seeded input, not model output]" if rel in seeded_set else ""
        lines.append(f"  {rel}{tag}")
        count += 1
    if truncated:
        lines.append(f"  ... (+ more, truncated at {max_files})")
    for ap in extra:
        lines.append(f"  {ap}   [written OUTSIDE the workspace by this run]")
    return "\n".join(lines) if lines else "  (workspace empty)"


def inline_files(workdir: str, extra: list[str] = (), max_files: int | None = None,
                 max_bytes: int | None = None, truncate: bool = False,
                 seeded: Iterable[str] = ()) -> str:
    """Inline the contents of text files under `workdir` (and `extra`). With max_files None
    (the report) every text file is inlined; the judge passes caps to keep its prompt cheap.
    A file over `max_bytes` is skipped by default or, with `truncate=True`, inlined up to the
    cap with a truncation note. Paths in `seeded` are labelled as pre-seeded inputs. Non-text
    files are skipped (they appear in `file_tree`).

    Every text file NOT inlined because a budget ran out is counted and named in a trailing
    note. Silence there is a correctness bug, not a cosmetic one: the judge grades with tools
    disabled, so a file dropped without a word is indistinguishable to it from a file the run
    never produced — and it fails the rubric item for a file that is sitting in workspace/."""
    seeded_set = set(seeded or ())
    chunks: list[str] = []
    used = 0
    over_cap: list[str] = []     # text files the file-count budget had no room for
    over_bytes: list[str] = []   # text files skipped whole for exceeding max_bytes

    def _candidates() -> Iterator[tuple[str, str]]:
        for ap, rel in _iter_files(workdir):
            yield ap, (f"{rel}  [seeded input, not model output]"
                       if rel in seeded_set else rel)
        for ap in extra:
            yield ap, f"{ap}  [outside workspace]"

    for path, label in _candidates():
        if not _is_text(path):
            continue          # binary: contents skipped, but file_tree still lists it
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if max_bytes is not None and size > max_bytes and not truncate:
            over_bytes.append(label)
            continue
        # Checked after the filters above so the count reflects files that would really
        # have been inlined, not every entry left in the walk.
        if max_files is not None and used >= max_files:
            over_cap.append(label)
            continue
        try:
            with open(path, "rb") as fh:
                raw = fh.read(max_bytes) if max_bytes is not None else fh.read()
        except OSError:
            continue
        # Read bytes and decode here rather than reading text with a cap: `max_bytes` is a
        # BYTE budget — it is compared against os.path.getsize() above and reported as bytes
        # in the note below — but a text-mode read(n) caps CHARACTERS, so any non-ASCII file
        # would overrun the cap (up to 4x on UTF-8) and the note would misstate what was kept.
        # errors="replace" also absorbs the codepoint the byte-exact cut may have split.
        body = raw.decode("utf-8", errors="replace")
        body = body.replace("\r\n", "\n").replace("\r", "\n")  # as text mode did
        if max_bytes is not None and size > max_bytes:
            body += (f"\n… [truncated at {max_bytes} bytes of {size} — "
                     "full file in workspace/]")
        chunks.append(f"--- {label} ---\n{body}")
        used += 1

    if over_cap:
        chunks.append(f"--- NOT INLINED: {len(over_cap)} more text file(s), over the "
                      f"{max_files}-file budget ---\n"
                      + "\n".join(f"  {n}" for n in over_cap)
                      + "\nTheir contents are absent from this view — do not read that as "
                        "the files being absent or empty.")
    if over_bytes:
        chunks.append(f"--- NOT INLINED: {len(over_bytes)} text file(s) larger than "
                      f"{max_bytes} bytes ---\n"
                      + "\n".join(f"  {n}" for n in over_bytes)
                      + "\nTheir contents are absent from this view — do not read that as "
                        "the files being absent or empty.")
    return "\n\n".join(chunks)
