"""Where q2/ keeps its data, its outputs, and its importable modules.

Every script under ``q2/`` starts with the same prologue::

    _PKG_DIR = _Path(__file__).resolve().parents[1]   # q2/
    _sys.path.insert(0, str(_PKG_DIR))
    import _q2_paths

so this module can be imported from any stage directory.  Sibling stages are
resolved by *filename stem*, which is why ``source_dirs(__file__)`` only needs
the calling script's path.

Paths are defined here and nowhere else; a script that needs the raw data or an
output directory imports it from this module.  The layout follows the
repository-wide convention (see the root ``README.md``)::

    <repo>/
      data/real_attachments/        raw competition attachments (not in git)
      results/                everything the scripts generate (not in git)
      q2/                           this package

Two locations can be moved with environment variables, which is useful when the
attachments live on a different drive::

    HUAWEI_DATA_DIR      replaces <repo>/data/real_attachments
    HUAWEI_RESULTS_DIR  replaces <repo>/results
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

def _normalised(path: Path) -> Path:
    """Lower-case the path on Windows, and keep it un-resolved.

    Two deliberate properties:

    * **No ``resolve()``.**  Python 3.9 absolutizes ``__file__`` before the
      script runs, so ``Path(__file__).parent`` is already anchored to this
      file; calling ``resolve()`` would only rewrite symlinks and re-case the
      path.  Dropping it keeps the constants in the form the caller used.
    * **Lower-case on Windows.**  The filesystem reports the on-disk case
      (``Q2``) while callers may type ``q2``; mixing the two made
      ``Path.relative_to`` fail, so the constants are canonicalised once here.
    """

    text = str(path)
    return Path(text.lower()) if os.name == "nt" else Path(text)


#: ``Q2/`` — this package.
PROJECT_ROOT = _normalised(Path(__file__).parent)
#: The repository root (``Q2/`` sits directly under it).
REPO_ROOT = _normalised(Path(__file__).parent.parent)

#: Stage directories, in execution order.
STAGE_DIRS = ("Q_2_1", "Q_2_2", "Q_2_3", "Q_2_4")


def _resolve(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return _normalised(Path(override).expanduser()) if override else default


#: Raw attachments (A: mixture/quality, B: scaling-law logs, C: leaderboards).
ATTACHMENTS = _resolve("HUAWEI_DATA_DIR", REPO_ROOT / "data" / "real_attachments")
A_DATA = ATTACHMENTS / "A_data_value"
B_DATA = ATTACHMENTS / "B_scaling_laws"
C_DATA = ATTACHMENTS / "C_efficiency_evolution"
REGMIX_TABLES = A_DATA / "regmix_tables"

#: Generated artifacts.
ANALYSIS_DIR = _resolve("HUAWEI_RESULTS_DIR", REPO_ROOT / "results")
#: Problem 1 hands its deliverables to problem 2 here.
HANDOFF_IN = ANALYSIS_DIR / "Q_1_to_2"
#: Problem 2 hands its deliverables to problem 3 here.
HANDOFF_OUT = ANALYSIS_DIR / "Q2_to_Q3"

#: Dataset A — per-domain losses and the domain-domain normalisation scales.
A_QUALITY_DESIGN = B_DATA / "supplementary_NQ_experiment_expanded.csv"  # named for its role, see README
#: Dataset B — the Pythia training log that identifies the classic law.
B_MAIN_LOG = B_DATA / "pythia_training_log_existing.csv"
#: Dataset B — Cerebras training log, the second classic-law source.
B_SECOND_LOG = B_DATA / "cerebras_training_log.csv"
#: Classic-law fit written by ``Q_2_1_3``; supplies the exponents reused by ``Q_2_2``.
CLASSIC_RESULTS = ANALYSIS_DIR / "scaling_law_full" / "scaling_law_full_results.json"


def repo_relative(path: str | Path) -> str:
    """Portable, forward-slashed path for reports and JSON payloads.

    Deliverables should not record where the repository happened to live on the
    machine that produced them, so anything written into a report or a JSON
    payload goes through here.  Paths outside the repository (an ``--input``
    pointing elsewhere) are kept as-is, because there is no shorter way to name
    them.
    """

    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def file_stem(path: str | Path) -> str:
    """``Q_2_1_2_classic_law_early_stopping.py`` -> ``classic_law_early_stopping``.

    The stage prefix (``Q_2_1_2_``) is stripped, because the scripts refer to
    each other by content name; ``directory_for`` adds the prefix back when it
    looks the file up.  Scripts also use this to reproduce their own invocation
    path, so a renamed file cannot drift from the command in its docstring.
    """

    stem = Path(path).name
    if stem.endswith(".py"):
        stem = stem[:-3]
    return re.sub(r"^Q_2_\d_\d_", "", stem)


def script_name(path: str | Path) -> str:
    """Repo-relative invocation path of the script ``path`` refers to."""

    resolved = Path(path).resolve()
    return f"q2/{resolved.parent.name}/{resolved.name}"


def directory_for(stem: str) -> str:
    """Stage directory that holds the module whose filename ends with ``stem``."""

    for directory in STAGE_DIRS:
        if list((PROJECT_ROOT / directory).glob(f"Q_2_*_{stem}.py")):
            return directory
    raise FileNotFoundError(f"no q2 script ending with _{stem}.py")


def sibling_imports(path: str | Path) -> list[str]:
    """Stage modules the script at ``path`` imports, by content name.

    Scripts import each other by content name (``Q_2_2_1_generalized_law``);
    reading them out of the source is what lets ``source_dirs`` find the right
    stage directory without every script hard-coding one.
    """

    source = Path(path).read_text(encoding="utf-8")
    found = re.findall(r"^(?:import|from)\s+(Q_2_\d_\d_(\w+))", source, flags=re.M)
    return [stem for _, stem in found]


def source_dirs(path: str | Path) -> list[Path]:
    """Directories that must be importable so this script's imports resolve.

    The calling script's own directory comes first (it holds the sibling stage
    modules); every *other* stage module the script imports is located by
    content name.  Appends ``results/Q2_to_Q3`` when present, because the
    evaluator that ``Q_2_3_1`` generates is imported from there.
    """

    directories: list[Path] = [
        Path(path).resolve().parent,
        PROJECT_ROOT,
    ]
    for stem in sibling_imports(path):
        try:
            directories.append(PROJECT_ROOT / directory_for(stem))
        except FileNotFoundError:
            continue
    if HANDOFF_OUT.is_dir():
        directories.append(HANDOFF_OUT)
    # Deduplicate while keeping the order above.
    unique: list[Path] = []
    for directory in directories:
        if directory not in unique:
            unique.append(directory)
    return unique


def install_source_dirs(path: str | Path) -> list[Path]:
    """Make ``source_dirs(path)`` importable, and return what was added.

    This is what the prologue in every script calls; ``source_dirs`` stays a
    pure query so it can be inspected without side effects.
    """

    directories = source_dirs(path)
    for directory in reversed(directories):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return directories
