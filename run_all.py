#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified entry point for the four problems in this repository.

Usage
-----
    python run_all.py --list              show every problem and its scripts
    python run_all.py q2                  run problem 2 only
    python run_all.py q1 q2               run problems 1 and 2, in that order
    python run_all.py --all               run every implemented problem
    python run_all.py q1 --python C:\\path\\to\\python.exe
    python run_all.py q2 --stop-on-error  abort at the first failing script

Each script runs from its own directory, in the order listed below, because the
stages are ordered by dependency: problem 1 produces the handoff that problem 2
reads, problem 2 produces the handoff that problem 3 needs, and problem 4 needs
problem 3's cost-allocation result.

Problems 1, 2 and 4 are driven stage by stage.  Problem 3 ships its own runner
(``Q3/run.py``) and is launched through it, so its internal layout stays free to
change; override the arguments with ``--q3-args``.

The per-problem ``run_all.ps1`` scripts are equivalent for a single problem; this
entry point exists so the whole repository can be driven from one command, from
any working directory, on any platform.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
#: Writable matplotlib cache, as the per-problem runners also arrange.
MPL_CACHE_DIR = REPO_ROOT / ".mplcfg"
#: Per-run script output lives here: ``log/<timestamp>/<problem>/<script>.log``.
LOG_ROOT = REPO_ROOT / "log"
#: Default WSL interpreter for problems that need POSIX (see --wsl-python).
DEFAULT_WSL_PYTHON = "~/miniconda3/envs/astroyd/bin/python"


def shell_quote(part: str) -> str:
    """Quote for ``bash -lc``, leaving a leading ``~`` unquoted.

    ``'~/miniconda3/.../python'`` is passed literally and cannot be found; bash
    only expands ``~`` when it is unquoted, so only the rest is quoted.
    """

    if part.startswith("~/"):
        return "~/" + shlex.quote(part[2:])
    return shlex.quote(part)


def to_posix(path: Path) -> str:
    """Windows path -> the WSL view of the same file (``D:\\a\\b`` -> ``/mnt/d/a/b``)."""

    text = str(path).replace("\\", "/")
    if len(text) > 1 and text[1] == ":":
        return f"/mnt/{text[0].lower()}{text[2:]}"
    return text


def probe_python(command: list[str], cwd: Path) -> tuple[int, int] | None:
    """Return ``(major, minor)`` for an interpreter, or None when unusable."""

    try:
        result = subprocess.run([*command, "-c", "import sys;print('%d %d' % sys.version_info[:2])"],
                                cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        major, minor = (int(part) for part in result.stdout.split()[:2])
    except ValueError:
        return None
    return major, minor


def choose_interpreter(requested: str, problem: Problem, wsl_python: str) -> tuple[list[str], str]:
    """Pick the command prefix used to launch ``problem``'s scripts.

    A POSIX-only problem is launched through WSL when this host has it and the
    requested interpreter cannot satisfy ``min_python``; otherwise the requested
    interpreter is used.  Returns the command prefix plus a short description.
    """

    if problem.needs_posix:
        wsl = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]
        version = probe_python(["wsl", "-d", "Ubuntu", "--", wsl_python], REPO_ROOT)
        if version is not None and version >= problem.min_python:
            return wsl, f"WSL {wsl_python} (Python {version[0]}.{version[1]})"
        version = probe_python([requested], REPO_ROOT)
        if version is not None and version >= problem.min_python:
            return [requested], f"{requested} (Python {version[0]}.{version[1]})"
    return [requested], requested


def run_command(
    argv: list[str],
    cwd: Path,
    dry_run: bool,
    log_path: Path | None = None,
    interpreter: list[str] | None = None,
) -> int:
    """Run one script, mirroring its stdout/stderr into ``log_path`` when given.

    ``interpreter`` is a command prefix (``["python"]`` or a WSL launch command)
    that ``argv`` is appended to.  The child is asked for UTF-8 through
    ``PYTHONIOENCODING`` so the log stays readable; if it ignores that, its bytes
    are decoded with the console encoding instead.
    """
    prefix = list(interpreter or [])
    call = [*prefix, *argv]
    printable = " ".join(shlex.quote(part) for part in call)
    print(f"run  {printable}", flush=True)
    if log_path is not None:
        print(f"     log -> {display_path(log_path)}", flush=True)
    if dry_run:
        return 0

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    if log_path is None:
        return subprocess.run(call, cwd=cwd, env=child_env).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as sink:
        sink.write(f"$ {printable}\n$ cwd={cwd}\n\n")
        sink.flush()
        captured = subprocess.run(
            call, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=False,
        )
        if captured.stdout is None:
            text = ""
        else:
            try:
                text = captured.stdout.decode("utf-8")
            except UnicodeDecodeError:
                text = captured.stdout.decode(recover_console_encoding(), "replace")
        sink.write(text)
        sink.write(f"\n$ exit={captured.returncode}\n")

    # Echo for a watcher, downgrading characters this console cannot show.
    encoding = console_encoding(sys.stdout)
    for line in text.splitlines():
        print(line.encode(encoding, "replace").decode(encoding, "replace"))
    return captured.returncode


@dataclass
class Problem:
    """One problem: either a list of stages, or a single runner script."""

    key: str
    title: str
    #: ``(stage label, [repository-relative script paths])``.  An entry may carry
    #: arguments after the path, e.g. ``"Q3/run.py --grid"``.
    stages: list[tuple[str, list[str]]] = field(default_factory=list)
    #: Repository-relative runner script, with default arguments.
    runner: tuple[str, list[str]] | None = None
    #: Prerequisites the scripts need but the repository does not ship.
    requires: list[str] = field(default_factory=list)
    note: str = ""
    #: Interpreter requirements beyond the default one.
    min_python: tuple[int, int] = (3, 9)
    #: Set when the problem cannot run under a native Windows interpreter.
    needs_posix: bool = False

    def all_scripts(self) -> list[str]:
        if self.runner is not None:
            return [self.runner[0]]
        return [entry.split(" ", 1)[0] for _, entries in self.stages for entry in entries]

    def missing_scripts(self) -> list[str]:
        return [script for script in self.all_scripts() if not (REPO_ROOT / script).is_file()]


#: Problems in execution order.  Paths are repository-relative; each script runs
#: from its own directory, matching the per-problem README.
PROBLEMS: tuple[Problem, ...] = (
    Problem(
        key="q1",
        title="问题一 语料质量、冲突消解与领域配比",
        stages=[
            # Order matters within this stage: the standardized 22-indicator
            # audit reads A1_frozen_scale.json / A1_directional_signals.csv from
            # _3_freeze_scale and A1_indicator_clusters.csv from _4_structure, so
            # it cannot run where its number suggests.
            ("Q1.1 质量表征", [
                "Q1/Q1_1/Q_1_1_1_audit_A1.py",
                "Q1/Q1_1/Q_1_1_2_scalarize_A1.py",
                "Q1/Q1_1/Q_1_1_3_freeze_scale.py",
                "Q1/Q1_1/Q_1_1_4_structure.py",
                "Q1/Q1_1/Q_1_1_3_audit_standardized_A1.py",
                "Q1/Q1_1/Q_1_1_5_score_A1.py",
                "Q1/Q1_1/Q_1_1_6_validate_extensions.py",
                "Q1/Q1_1/Q_1_1_7_blind_text_review.py",
                "Q1/Q1_1/Q_1_1_7b_ai_review_complete.py",
                "Q1/Q1_1/Q_1_1_8_baselines.py",
                "Q1/Q1_1/Q_1_1_9_ai_text_audit.py",
                "Q1/Q1_1/Q_1_1_10_verify_outputs.py",
                "Q1/Q1_1/Q_1_1_11_sensitivity_A1.py",
                "Q1/Q1_1/Q_1_1_12_direction_evidence.py",
            ]),
            ("Q1.2 冲突消解", [
                "Q1/Q1_2/Q_1_2_1_prepare_signals.py",
                "Q1/Q1_2/Q_1_2_2_relation_graph.py",
                "Q1/Q1_2/Q_1_2_3_crossfit_residuals.py",
                "Q1/Q1_2/Q_1_2_4_conflict_scores.py",
                "Q1/Q1_2/Q_1_2_5_cause_analysis.py",
                "Q1/Q1_2/Q_1_2_6_reliability_and_Qstar.py",
                "Q1/Q1_2/Q_1_2_7_baselines.py",
                "Q1/Q1_2/Q_1_2_8_sensitivity.py",
                "Q1/Q1_2/Q_1_2_9_text_cases.py",
                "Q1/Q1_2/Q_1_2_11_tail_evidence.py",
                "Q1/Q1_2/Q_1_2_10_verify_outputs.py",
            ]),
            ("Q1.3 领域配比", [
                "Q1/Q1_3/Q_1_3_1_load_and_standardize.py",
                "Q1/Q1_3/Q_1_3_2_scheffe_features.py",
                "Q1/Q1_3/Q_1_3_3_fit_scheffe_models.py",
                "Q1/Q1_3/Q_1_3_4_quality_prior.py",
                "Q1/Q1_3/Q_1_3_5_substitution_complement.py",
                "Q1/Q1_3/Q_1_3_6_validation.py",
                "Q1/Q1_3/Q_1_3_7_bootstrap_uncertainty.py",
                "Q1/Q1_3/Q_1_3_10_evidence_checks.py",
                "Q1/Q1_3/Q_1_3_11_final_checks.py",
                "Q1/Q1_3/Q_1_3_13_experiment_manifest.py",
                "Q1/Q1_3/Q_1_3_12_handoff.py",
                "Q1/Q1_3/Q_1_3_9_verify_outputs.py",
            ]),
        ],
        requires=[
            "data/real_attachments/A_data_value/",
            "data/real_attachments/B_scaling_laws/",
        ],
        note="产出 results/Q_1_to_2/ 供问题二读取。",
    ),
    Problem(
        key="q2",
        title="问题二 经典与广义标度律、边际效用与替代",
        stages=[
            ("Q2.1 经典标度律", [
                "Q2/Q_2_1/Q_2_1_1_classic_law.py",
                "Q2/Q_2_1/Q_2_1_2_classic_law_early_stopping.py",
                "Q2/Q_2_1/Q_2_1_3_classic_law_analysis.py",
                "Q2/Q_2_1/Q_2_1_4_classic_law_figures_zh.py",
            ]),
            ("Q2.2 广义标度律（质量 + 配比）", [
                "Q2/Q_2_2/Q_2_2_1_generalized_law.py",
            ]),
            ("Q2.3 交付第三问", [
                # Q_2_3_1 writes the handoff payload that Q_2_2_2 draws
                # (mixture_response.json), so the figures run after it.
                "Q2/Q_2_3/Q_2_3_1_handoff_build.py",
                "Q2/Q_2_2/Q_2_2_2_generalized_law_figures.py",
                "Q2/Q_2_3/Q_2_3_2_handoff_paper_form.py",
                "Q2/Q_2_3/Q_2_3_3_handoff_identification.py",
            ]),
            ("Q2.4 边际效益、弹性与替代", [
                "Q2/Q_2_4/Q_2_4_1_elasticity_analysis.py",
                "Q2/Q_2_4/Q_2_4_2_elasticity_figures.py",
                "Q2/Q_2_4/Q_2_4_3_factor_elasticity_analysis.py",
                "Q2/Q_2_4/Q_2_4_4_factor_elasticity_figures.py",
                "Q2/Q_2_4/Q_2_4_5_quality_substitution_analysis.py",
                "Q2/Q_2_4/Q_2_4_6_quality_substitution_figures.py",
                "Q2/Q_2_4/Q_2_4_7_joint_substitution_analysis.py",
                "Q2/Q_2_4/Q_2_4_8_joint_substitution_figures.py",
            ]),
        ],
        requires=[
            "data/real_attachments/A_data_value/regmix_tables/",
            "data/real_attachments/B_scaling_laws/",
            "results/Q_1_to_2/  （问题一的交付物）",
        ],
        note="Q2.4 依赖 Q2.3 生成的 generalized_law_evaluator.py，顺序不能颠倒。",
    ),
    Problem(
        key="q3",
        title="问题三 算力预算下的资源配置与质量投入",
        # Problem 3 is a self-contained package with its own entry point, run
        # stage by stage here so the two handoff bridges are part of the chain:
        # sync pulls the fresh Q1/Q2 outputs into the copies Q3 reads, then the
        # grid runs, then the anchors are exported for problem 4.
        stages=[
            ("Q3.0 同步问题一、二的交接文件", [
                "Q3/Q3_1/sync_q3_handoff.py",
            ]),
            ("Q3.1 全部算力档扫描", [
                "Q3/run.py --grid --output results/full_grid.jsonl",
            ]),
            ("Q3.2 导出问题四所需的锚点表", [
                "Q3/Q3_3/write_q4_anchors.py --input Q3/results/full_grid.jsonl",
            ]),
        ],
        requires=[
            "results/Q_1_to_2/  （问题一的交付物）",
            "results/Q2_to_Q3/  （问题二的交付物）",
        ],
        note=(
            "问题三需要 Python 3.10+（使用 X | None 注解）。本机若装了 WSL，"
            "本入口会自动改用 WSL 里的解释器（见 --wsl-python）；"
            "问题三自带交接输入副本，Q3.0 会把本次 Q1/Q2 的产物同步进去。"
        ),
        min_python=(3, 10),
        needs_posix=True,
    ),
    Problem(
        key="q4",
        title="问题四 能力前沿、Loss--Benchmark 桥接与预测",
        stages=[
            ("Q4.1 数据准备", [
                "Q4/Q4_1/Q_4_1_1_rebuild_C8.py",
                "Q4/Q4_1/Q_4_1_2_match_C1_C4.py",
                "Q4/Q4_1/Q_4_1_3_c6_bridge_plan.py",
            ]),
            ("Q4.2 统一状态模型", [
                "Q4/Q4_2/Q_4_2_1_sample_and_frontier.py",
                "Q4/Q4_2/Q_4_2_2_compute_frontier.py",
                "Q4/Q4_2/Q_4_2_3_bridge_and_mechanism.py",
                "Q4/Q4_2/Q_4_2_4_forecast.py",
                "Q4/Q4_2/Q_4_2_5_validation.py",
                "Q4/Q4_2/Q_4_2_7_verify_outputs.py",
            ]),
        ],
        requires=[
            "data/real_attachments/C_efficiency_evolution/",
            "data/real_attachments/C_efficiency_evolution/detailed_results/  （或原始 F题.zip）",
            "问题三的成本分配结果（Q4/Q4_2/q4_utils.py 顶部的 Q3_RESULT 常量）",
        ],
        note="问题三结果未就位前，问题四的完整流程无法从零运行。",
    ),
)


def by_key(key: str) -> Problem | None:
    for problem in PROBLEMS:
        if problem.key == key.lower():
            return problem
    return None


def configure_environment() -> None:
    """Give matplotlib a writable cache directory, as the ps1 runners do."""

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))


def print_listing() -> int:
    for problem in PROBLEMS:
        scripts = problem.all_scripts()
        print(f"\n{problem.key}  {problem.title}")
        if problem.runner is not None:
            runner, runner_args = problem.runner
            mark = " " if (REPO_ROOT / runner).is_file() else "!"
            print(f"  [runner] {mark} {runner} {' '.join(runner_args)}".rstrip())
        for stage, stage_scripts in problem.stages:
            print(f"  [{stage}]")
            for script in stage_scripts:
                mark = " " if (REPO_ROOT / script).is_file() else "!"
                print(f"   {mark} {script}")
        if problem.requires:
            print("  requires:")
            for item in problem.requires:
                print(f"    - {item}")
        if problem.note:
            print(f"  note: {problem.note}")
    print("\nmarker: '!' = file not found")
    return 0


def display_path(path: Path) -> str:
    """Repo-relative path when possible, otherwise the path as given.

    Callers may hand over either an absolute or a relative path (``run_all.py``
    builds absolute ones, a test harness may not), and ``Path.relative_to``
    refuses to mix the two, so fall back rather than raise.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def console_encoding(stream) -> str:
    """Encoding this console can actually print.

    Captured child output is UTF-8 once ``PYTHONIOENCODING`` is set below, but a
    cp936 console cannot encode every character it contains; echoing it raw
    would raise ``UnicodeEncodeError`` and abort the run.
    """
    return getattr(stream, "encoding", None) or "utf-8"


def recover_console_encoding() -> str:
    """Best guess at the encoding a child process writes with by default."""
    if os.name != "nt":
        import locale

        return locale.getpreferredencoding(False) or "utf-8"
    try:
        import ctypes

        return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    except Exception:
        return "utf-8"


def run_problem(
    problem: Problem,
    python: str,
    stop_on_error: bool,
    dry_run: bool,
    q3_args: str = "",
    log_dir: Path | None = None,
    wsl_python: str = DEFAULT_WSL_PYTHON,
) -> list[str]:
    """Run one problem; return the repository-relative failures."""

    missing = problem.missing_scripts()
    if missing:
        print(f"!! {problem.key}: 缺少脚本文件：")
        for script in missing:
            print(f"     {script}")

    interpreter, label = choose_interpreter(python, problem, wsl_python)
    print(f"     解释器: {label}")

    def log_for(script: str) -> Path | None:
        if log_dir is None:
            return None
        # Mirror the layout under log/ and keep the name recognisable.
        return log_dir / problem.key / f"{Path(script).stem}.log"

    def launch(path: Path, args: list[str], script: str) -> int:
        """Build the argv for one script, translating to a WSL view when needed."""

        if interpreter[:1] == ["wsl"]:
            command = (
                f"cd {shlex.quote(to_posix(path.parent))} && "
                + " ".join(shell_quote(part) for part in [wsl_python, path.name, *args])
            )
            return run_command([*interpreter, command], path.parent, dry_run,
                               log_for(script), interpreter=[])
        return run_command([path.name, *args], path.parent, dry_run,
                           log_for(script), interpreter=interpreter)

    failures: list[str] = []
    if problem.runner is not None:
        runner, default_args = problem.runner
        path = REPO_ROOT / runner
        if not path.is_file():
            print(f"skip {runner} (not found)")
            return [runner]
        args = shlex.split(q3_args) if q3_args else default_args
        print(f"\n== [{problem.key}] {problem.title} ==")
        code = launch(path, args, runner)
        if code != 0:
            print(f"FAIL {runner} (exit code {code})")
            failures.append(runner)
        else:
            print(f"ok   {runner}")
        return failures

    for stage, stage_scripts in problem.stages:
        print(f"\n== [{problem.key}] {stage} ==")
        for entry in stage_scripts:
            # A stage entry may carry arguments after the script path, e.g.
            # "Q3/run.py --grid --output results/full_grid.jsonl".
            parts = shlex.split(entry)
            script, script_args = parts[0], parts[1:]
            path = REPO_ROOT / script
            if not path.is_file():
                print(f"skip {script} (not found)")
                failures.append(script)
                if stop_on_error:
                    return failures
                continue
            code = launch(path, script_args, script)
            if code != 0:
                print(f"FAIL {entry} (exit code {code})")
                failures.append(script)
                if stop_on_error:
                    return failures
            else:
                print(f"ok   {entry}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按依赖顺序运行本仓库各题目的脚本。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run_all.py --list\n"
            "  python run_all.py q2\n"
            "  python run_all.py q1 q2 --stop-on-error\n"
            '  python run_all.py q3 --q3-args "--budget 1e22 --context 4096"\n'
            "  python run_all.py --all\n"
        ),
    )
    parser.add_argument("problems", nargs="*", help="题目编号，如 q1 q2 q3 q4")
    parser.add_argument("--all", action="store_true", help="运行全部题目")
    parser.add_argument("--list", action="store_true", help="只列出脚本与前置条件")
    parser.add_argument("--python", default=sys.executable, help="用于运行脚本的解释器（默认当前解释器）")
    parser.add_argument("--stop-on-error", action="store_true", help="某个脚本失败后立即停止")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    parser.add_argument("--no-logs", action="store_true",
                        help=f"不写日志（默认写入 {LOG_ROOT.name}/<时间戳>/<题目>/<脚本>.log）")
    parser.add_argument("--q3-args", default="", help="覆盖问题三的入口参数（默认 --grid）")
    parser.add_argument("--wsl-python", default=DEFAULT_WSL_PYTHON,
                        help="需要 POSIX 的题目所用的 WSL 解释器路径")
    args = parser.parse_args(argv)

    if args.list:
        return print_listing()

    if args.all:
        selected = list(PROBLEMS)
    elif args.problems:
        selected = []
        for key in args.problems:
            problem = by_key(key)
            if problem is None:
                parser.error(f"未知题目 {key!r}；可选：q1 q2 q3 q4")
            selected.append(problem)
    else:
        parser.print_help()
        print("\n请指定题目，或使用 --all / --list。")
        return 2

    configure_environment()
    print(f"解释器: {args.python}")
    print(f"仓库根目录: {REPO_ROOT}")

    # Every run keeps its per-script output under log/<timestamp>/<problem>/.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = None if args.no_logs else LOG_ROOT / stamp
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"日志目录: {display_path(log_dir)}")

    failures: list[str] = []
    for problem in selected:
        failures.extend(
            run_problem(problem, args.python, args.stop_on_error, args.dry_run,
                        args.q3_args, log_dir, args.wsl_python)
        )

    print("\n" + "=" * 72)
    if log_dir is not None:
        print(f"各脚本的完整输出见 {display_path(log_dir)}/")
    if failures:
        print(f"完成，但有 {len(failures)} 个脚本失败：")
        for script in failures:
            print(f"  {script}")
        return 1
    print("全部脚本执行成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
