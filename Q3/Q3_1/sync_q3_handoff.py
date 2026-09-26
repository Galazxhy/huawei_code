#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish problems 1 and 2 outputs into the handoff copies problem 3 reads.

Problem 3 is self-contained: it reads everything from
``Q3/Q3_1/00_交接输入/`` so it can be run on a machine that has no attachment
tree.  Those copies are a snapshot, though, and a fresh ``Q1 -> Q2`` run
produces *new* files with new hashes, which problem 3 records in its source
manifest.  This script is the bridge: it copies the current handoff outputs
into the copies, so a from-scratch ``Q1 -> Q2 -> Q3`` run is internally
consistent.

Run from the repository root (a new Q2 output must exist first):

    python Q3/Q3_1/sync_q3_handoff.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upwards until the repository root is found.

    The depth of this file inside Q3 is an implementation detail that has
    already changed once; locating the root by a marker file is stable.
    """

    for candidate in (start, *start.parents):
        if (candidate / "run_all.py").is_file() and (candidate / "Q2").is_dir():
            return candidate
    raise FileNotFoundError(f"找不到仓库根目录（从 {start} 向上查找 run_all.py 失败）")


HERE = Path(__file__).resolve().parent
REPO = find_repo_root(HERE)
INPUTS = HERE / "00_交接输入"

#: (source directory, destination under 00_交接输入)
#: Note the destination names: problem 3 reads ``Q1_to_Q2`` and ``Q2_to_Q3``,
#: while problem 2 writes into ``Q_1_to_2`` (with the underscore) and
#: ``Q2_to_Q3``.  The two spellings are deliberate on problem 3's side.
PAIRS = (
    (REPO / "results" / "Q_1_to_2", INPUTS / "Q1_to_Q2"),
    (REPO / "results" / "Q2_to_Q3", INPUTS / "Q2_to_Q3"),
)


def publish() -> list[str]:
    copied: list[str] = []
    for source, destination in PAIRS:
        if not source.is_dir():
            raise FileNotFoundError(
                f"找不到 {source}；请先按顺序跑完问题一、问题二，再运行本脚本。"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.is_dir() or item.suffix == ".pyc":
                continue
            shutil.copy2(item, destination / item.name)
            copied.append(f"{source.name}/{item.name}")
    return copied


def main() -> int:
    copied = publish()
    print(f"已同步 {len(copied)} 个交接文件到 {INPUTS}:")
    for name in copied:
        print("  ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
