"""Merge several runs' artifacts into one summary.json + summary.md.

WHY THIS EXISTS
---------------
A sweep can be run as one process or as several. `--jobs` threads the cells inside one
harness process and needs nothing from this module: one process writes one run directory and
one summary, exactly as a serial run does. A fan-out — N harness processes, each with a shard
of the evals — cannot, because a run directory is written by the process that owns it. What
comes back is N summaries of a sweep nobody asked for in N pieces, and the question the sweep
was run to answer ("did this skill regress") has to be reassembled by eye from all of them.

So this reassembles it, into artifacts that read like the single-process ones: a summary.json
whose `cells` are every shard's cells and whose counts are over all of them, and a summary.md
with one results table. Downstream tooling — Format-EvalSummary.ps1, Add-RunProvenance.ps1,
anything reading summary.json — then works on a fan-out without knowing it was one.

WHAT IS COPIED AND WHAT IS RECOMPUTED
-------------------------------------
The markdown rows are copied VERBATIM from the shards' own summary.md, with only the report
link repointed at the shard's subdirectory. Nothing about a cell's text is regenerated here:
the mark, the det/rubric split, the costs and any field a later harness adds all pass through
untouched. They could not be regenerated faithfully in any case — the rubric split comes from
the judge's assertions, which summary.json does not carry — and a second renderer that got it
subtly wrong would produce a merged summary that disagreed with the shard summaries beside it.

The numbers, by contrast, are recomputed from the shards' summary.json, which is the
authoritative record of pass/fail. The pass-rate table is derived the way `render_markdown`
derives it (ungraded cells out of the numerator AND the denominator) rather than by counting
check marks in the copied text.

`consistency` is neither copied nor rebuilt from cells: it is re-judged across the union by
`runner.merge_consistency`. Two shards can each be internally uniform and still disagree with
each other, which is the drift that matters most in a fan-out and the one a conjunction of
their verdicts would report as `verified`. See that function for why it reads the published
spreads rather than the per-cell fields.

WHAT IT REFUSES
---------------
Shards that are not columns of the same matrix. Different agents, or different model targets,
mean the rows carry a different number of cells and in a different order — concatenating them
would produce a table whose columns are mislabelled rather than an error, and every number
read out of it afterwards would be wrong about which model earned it. Refused rather than
merged, and named in the message.
"""

from __future__ import annotations

import json
import os
import re

from .runner import merge_consistency

# A results table is the one whose first column is `eval`; the pass-rate table below it is
# keyed by model. Matched the way tools/eval-harness/Format-EvalSummary.ps1 matches it, because
# that script re-reads what this writes and the two have to agree on what a table is.
_EVAL_HEADER = re.compile(r"^\s*\|\s*eval\s*\|")
_LINK = re.compile(r"\]\((?!https?://)([^)]+)\)")


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_shards(run_dir: str) -> list[str]:
    """Immediate subdirectories of `run_dir` that hold a finished run, by name.

    Sorted, because the shard names carry their index (`shard-03of11`) and a merged file whose
    rows arrive in readdir order would differ between two merges of the same run.

    A shard with no summary.json is skipped rather than refused: a fan-out reports a shard that
    died on its own exit code and log, and refusing the merge would take the other ten shards'
    results down with it — which is the outcome sharding exists to avoid.
    """
    if not os.path.isdir(run_dir):
        return []
    return sorted(
        name for name in os.listdir(run_dir)
        if os.path.isfile(os.path.join(run_dir, name, "summary.json"))
    )


def _table_rows(markdown: str) -> tuple[str | None, list[str]]:
    """The results header and every data row, from a summary.md in either shape.

    Either shape matters: Format-EvalSummary.ps1 may already have rewritten a shard's summary
    into one table per eval suite, so the rows are collected from EVERY results table in the
    file rather than from the first one. That is the same pass that script makes, for the same
    reason, and it is what lets a merge run after the per-shard grouping instead of before it.
    """
    lines = markdown.splitlines()
    header = None
    rows: list[str] = []
    i = 0
    while i < len(lines):
        if not _EVAL_HEADER.match(lines[i]):
            i += 1
            continue
        if header is None:
            header = lines[i].strip()
        i += 1
        if i < len(lines) and lines[i].lstrip().startswith("|"):
            i += 1                                   # the |---|---| rule
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            rows.append(lines[i].strip())
            i += 1
    return header, rows


def _reroot(row: str, prefix: str) -> str:
    """Repoint a row's report links at the shard subdirectory they now live under.

    Every link in a summary.md is relative to the run directory that wrote it. Merged one
    level up, `claude-haiku-4-5/<suite>/<eval>/report.md` has to become
    `<shard>/claude-haiku-4-5/<suite>/<eval>/report.md` or every cell in the merged table
    links to a file that is not there — which is worse than no link, because it looks like one.

    Only relative targets are touched. Prefixing an absolute URL would corrupt it, and the
    harness has no reason to write one today, so this is a guard rather than a feature.
    """
    return _LINK.sub(lambda m: "](%s/%s)" % (prefix, m.group(1)), row)


def _target_key(target: dict) -> tuple:
    return (target.get("model"), target.get("reasoning_effort"))


def _target_label(target: dict) -> str:
    """`model@effort` when an effort is pinned — the column identity `ModelTarget.label` uses."""
    base = target.get("model") or "default"
    effort = target.get("reasoning_effort")
    return f"{base}@{effort}" if effort else base


def merge_runs(run_dir: str, shards: list[str] | None = None) -> dict:
    """Write `<run_dir>/summary.json` and `<run_dir>/summary.md` over the shards' own.

    Returns the merged summary dict. Raises ValueError with the disagreement named when the
    shards are not columns of one matrix.
    """
    names = shards if shards is not None else find_shards(run_dir)
    if not names:
        raise ValueError(f"no shard summaries under {run_dir} — nothing to merge")

    summaries = [(name, _read_json(os.path.join(run_dir, name, "summary.json")))
                 for name in names]

    # The refusals. Both are about the TABLE: rows are copied verbatim, so a row's cells are
    # positional against the header its own shard wrote, and a mismatch here does not fail —
    # it silently files one model's result under another's name.
    agents = sorted({s.get("agent") for _, s in summaries})
    if len(agents) > 1:
        raise ValueError(
            "refusing to merge shards that ran different agents: "
            + ", ".join(str(a) for a in agents)
            + ". Their rows carry different cells, so one table over both would label the "
              "columns wrongly rather than fail."
        )

    target_sets = {tuple(_target_key(t) for t in (s.get("targets") or [])): name
                   for name, s in summaries}
    if len(target_sets) > 1:
        detail = "; ".join(
            f"{name}: " + (", ".join(f"{m or 'default'}@{e}" if e else str(m or "default")
                                     for m, e in keys) or "(none)")
            for keys, name in sorted(target_sets.items(), key=lambda kv: kv[1])
        )
        raise ValueError(
            "refusing to merge shards with different model targets — their tables have "
            f"different columns: {detail}"
        )

    agent = agents[0]
    targets = summaries[0][1].get("targets") or []

    cells: list[dict] = []
    for name, summary in summaries:
        for cell in summary.get("cells") or []:
            cell = dict(cell)
            # Relative to the MERGED run directory now, matching the markdown links. Left
            # alone when the shard did not record one, rather than invented.
            if cell.get("artifacts"):
                cell["artifacts"] = f"{name}/{cell['artifacts']}"
            cell["shard"] = name
            cells.append(cell)

    # Judge identity is a property of how the cells were graded, so a fan-out that somehow
    # graded with two judges must not claim one. Reported as None rather than refused: unlike
    # the columns, this misnames nothing in the table.
    judge_agents = {s.get("judge_agent") for _, s in summaries}
    judge_models = {s.get("judge_model") for _, s in summaries}

    n_evals = sum(int(s.get("n_evals") or 0) for _, s in summaries)
    merged = {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "consistency": merge_consistency([s.get("consistency") or {} for _, s in summaries]),
        "command": (f"run --evals {n_evals} file(s) --agent {agent} "
                    f"(merged from {len(summaries)} shards)"),
        "agent": agent,
        "models": [t.get("model") for t in targets if t.get("model") is not None] or ["default"],
        "targets": targets,
        # The requested mode, and true of the merge only if it was true of every shard.
        "isolated": all(bool(s.get("isolated")) for _, s in summaries),
        "isolation_requested": all(bool(s.get("isolation_requested")) for _, s in summaries),
        "all_cells_isolated": all(bool(s.get("all_cells_isolated")) for _, s in summaries),
        "n_evals": n_evals,
        "n_cells": len(cells),
        "n_passed": sum(1 for c in cells if c.get("passed")),
        "judge_agent": judge_agents.pop() if len(judge_agents) == 1 else None,
        "judge_model": judge_models.pop() if len(judge_models) == 1 else None,
        # What this merge was made of, so a reader of the merged file can get back to the
        # shard that produced any given cell without matching names by hand.
        "shards": [
            {"run_id": name, "n_cells": int(s.get("n_cells") or 0),
             "n_passed": int(s.get("n_passed") or 0), "command": s.get("command")}
            for name, s in summaries
        ],
        "cells": cells,
    }

    header = None
    rows: list[str] = []
    for name, _ in summaries:
        path = os.path.join(run_dir, name, "summary.md")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            shard_header, shard_rows = _table_rows(fh.read())
        if header is None:
            header = shard_header
        rows.extend(_reroot(row, name) for row in shard_rows)

    labels = [_target_label(t) for t in targets]
    if header is None:
        header = "| eval | " + " | ".join(labels) + " |"
    rule = "|" + "---|" * (len(labels) + 1)

    # Sorted by eval name, which is what the harness's own renderer does. The rows arrive
    # grouped by shard, and Format-EvalSummary.ps1 regroups them by suite afterwards anyway,
    # but a merged file that is readable BEFORE that pass is worth the one sort.
    rows.sort(key=lambda r: r.strip().strip("|").split("|")[0].strip())

    lines = [
        f"# Eval results — {agent} ({merged['command']})",
        "",
        ("Each cell links to a per-cell `report.md` — the prompt the model was given, its "
         "complete response (full transcript), and the workspace files after the run "
         "(seeded inputs marked)."),
        "",
        (f"Merged from {len(summaries)} shard(s) run in parallel; each cell's link points "
         "into the shard that produced it."),
        "",
        header,
        rule,
    ]
    lines.extend(rows)

    lines += ["", "## Pass rate", "", "| model | pass rate |", "|---|---|"]
    for target in targets:
        key = _target_key(target)
        mine = [c for c in cells
                if (c.get("model"), c.get("reasoning_effort")) == key]
        graded = [c for c in mine if not c.get("ungraded")]
        n_pass = sum(1 for c in graded if c.get("passed"))
        ungraded = len(mine) - len(graded)
        rate = f"{n_pass}/{len(graded)}" + (f" ({ungraded} ungraded)" if ungraded else "")
        lines.append(f"| {_target_label(target)} | {rate} |")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return merged
