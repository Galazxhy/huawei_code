#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert problem 3's run records into the anchor table problem 4 reads.

Problem 4's unified state equation needs one anchor loss per budget, and it
opens the file named by ``Q4/Q4_2/q4_utils.py`` (``Q3_RESULT``).  Problem 3
writes JSONL run records over a wide budget grid; this script selects the rows
problem 4 filters on and flattens them into the CSV it expects.

The budget list, scenario/context/cost-form/mode selection and the resulting
validation all live in ``q4_utils``, which is imported here so the two sides
cannot drift apart.

    python Q3/Q3_3/write_q4_anchors.py                      # results/full_grid.jsonl
    python Q3/Q3_3/write_q4_anchors.py --input results/other.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upwards until the repository root is found (marker: run_all.py)."""

    for candidate in (start, *start.parents):
        if (candidate / "run_all.py").is_file() and (candidate / "Q2").is_dir():
            return candidate
    raise FileNotFoundError(f"找不到仓库根目录（从 {start} 向上查找 run_all.py 失败）")


HERE = Path(__file__).resolve().parent
Q3_ROOT = HERE.parent
REPO = find_repo_root(HERE)
Q4_UTILS = REPO / "Q4" / "Q4_2"

DEFAULT_INPUT = Q3_ROOT / "results" / "full_grid.jsonl"
DEFAULT_OUTPUT = REPO / "results" / "Q3" / "result_q3_cost_allocation.csv"

COLUMNS = ("scenario", "context", "cost_form", "mode", "bridge", "kappa", "budget", "loss")


def anchor_contract() -> tuple[list[float], list[dict[str, object]]]:
    """Read the selection contract from problem 4's utilities.

    Importing ``q4_utils`` is not an option -- it validates the anchor file at
    import time, which is the file this script writes -- so the budget constant
    is read out of the source without executing it.

    Problem 4 loads one anchor set per cost form (``exponential`` for the
    baseline, plus ``power`` and ``log_asymptotic`` for the mechanism and
    forecast comparisons), so every form is exported.
    """

    source = (Q4_UTILS / "q4_utils.py").read_text(encoding="utf-8")
    budgets: list[float] | None = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "Q3_ANCHOR_BUDGETS"
            for target in node.targets
        ):
            budgets = [float(value) for value in ast.literal_eval(node.value)]
            break
    if budgets is None:
        raise SystemExit(f"{Q4_UTILS / 'q4_utils.py'} 里找不到 Q3_ANCHOR_BUDGETS 常量")

    shared = {
        "scenario": "main",
        "context": 4096,
        "mode": "joint_reduced_hull",
        "bridge": "ratio",
        "kappa": 1.0,
    }
    selections = [{**shared, "cost_form": form}
                  for form in ("exponential", "power", "log_asymptotic")]
    return budgets, selections


def flatten(record: dict) -> dict | None:
    """One JSONL record -> one anchor row, or None when it carries no result."""

    job = record.get("job") or {}
    result = record.get("result")
    if not isinstance(result, dict) or record.get("status") != "PASS":
        return None
    return {
        "scenario": job.get("scenario"),
        "context": job.get("context"),
        "cost_form": job.get("form"),
        "mode": job.get("mode"),
        "bridge": job.get("bridge"),
        "kappa": job.get("kappa"),
        "budget": job.get("budget"),
        "loss": result.get("mean_loss"),
    }


def matches(row: dict, want: dict[str, object], budgets: list[float]) -> bool:
    for key, expected in want.items():
        actual = row.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12 * max(1.0, abs(expected)):
                return False
        elif str(actual) != str(expected):
            return False
    budget = row.get("budget")
    return budget is not None and any(abs(float(budget) - b) <= 1e-12 * max(1.0, abs(b)) for b in budgets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Q3 JSONL run records")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="anchor CSV for Q4")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"找不到 {args.input}；请先运行 python Q3/run.py --grid（或指定 --input）。")
        return 2

    budgets, selections = anchor_contract()
    rows: list[dict] = []
    total = 0
    records = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        row = flatten(json.loads(line))
        if row is not None:
            records.append(row)

    for want in selections:
        matched = [row for row in records if matches(row, want, budgets)]
        missing = [b for b in budgets
                   if not any(abs(float(row["budget"]) - b) <= 1e-12 * max(1.0, abs(b)) for row in matched)]
        if missing:
            raise SystemExit(
                f"{args.input} 中缺少问题四所需的预算档 {missing}（成本函数 {want['cost_form']}）；"
                f"选择条件是 {want}。"
            )
        rows.extend(matched)

    rows.sort(key=lambda row: (str(row["cost_form"]), float(row["budget"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"读取 {total} 条记录，写入 {args.output}：{len(rows)} 行锚点"
          f"（{len(selections)} 个成本函数 × {len(budgets)} 个预算档）")
    for row in rows:
        print(f"  {row['cost_form']:16} budget={row['budget']:.3g}  loss={row['loss']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
