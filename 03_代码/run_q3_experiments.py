"""Preflight and resumable registered Q3 experiments; writes only in PZ_Q3."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy

from q3_model_contract import ROOT, Q3ModelContract
from q3_optimizer import JointOptimizer, OptimizerConfig

MODES = ("joint_reduced_hull", "fixed_p", "fixed_q", "fixed_p_q_baseline")
FORMS = ("exponential", "power", "log_asymptotic")
BUDGETS = sorted([10. ** k for k in range(19, 25)] + [3 * 10. ** k for k in range(19, 24)])


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def scenarios(batch):
    base = {"bridge": "ratio", "kappa": 1., "config": asdict(OptimizerConfig())}
    if batch == "main":
        return [{**base, "scenario": "main"}]
    if batch == "negative":
        return [{**base, "kappa": 0., "scenario": "zero_quality_effect"}]
    if batch == "bridge":
        q0s = [None, float(Q3ModelContract().support_points.mean(axis=0) @ Q3ModelContract().q_i)]
        return [{"bridge": bridge, "kappa": k, "scenario": f"{bridge}_k{k}_q0{q0}",
                 "config": {**base["config"], "q0_override": q0}}
                for bridge, k, q0 in itertools.product(("ratio", "additive"), (.5, 1., 1.5), q0s)]
    if batch == "cost":
        return [{**base, "scenario": "normalized_cost", "config": {
            **base["config"], "normalize_cost_endpoint": True}}]
    if batch == "search":
        variants = [
            ("anchors48", {"anchor_count": 48}), ("anchors96", {"anchor_count": 96}),
            ("starts6", {"starts": 6}), ("seed2", {"seed": 20260925}),
            ("upper_half", {"n_bounds": [1e6, 5e10], "d_bounds": [1e9, 5e12]}),
            ("upper_double", {"n_bounds": [1e6, 2e11], "d_bounds": [1e9, 2e13]}),
            ("lower_tenth", {"n_bounds": [1e5, 1e11], "d_bounds": [1e8, 1e13]}),
        ]
        return [{**base, "scenario": label, "config": {**base["config"], **changes}}
                for label, changes in variants]
    raise ValueError("unknown batch")


def registered_jobs(batch, budgets=None):
    model = Q3ModelContract()
    for scenario, budget, context, form, mode in itertools.product(
            scenarios(batch), BUDGETS if budgets is None else budgets, model.contexts, FORMS, MODES):
        yield {**scenario, "budget": budget, "context": context, "form": form, "mode": mode}


def job_key(job):
    return json.dumps({k: v for k, v in job.items() if k != "scenario"}, sort_keys=True)


def job_id(job):
    return hashlib.sha256(job_key(job).encode()).hexdigest()[:20]


def registry_payload(batch, jobs, manifest):
    payload = json.loads(json.dumps({"batch": batch, "jobs": jobs, "source_manifest": manifest}))
    payload["registry_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def load_registry(path, manifest):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_manifest") != manifest:
        raise RuntimeError(f"stale or unversioned registry: {path}")
    expected = registry_payload(payload["batch"], payload["jobs"], manifest)
    if expected != payload:
        raise RuntimeError(f"registry content/hash mismatch: {path}")
    if not payload["jobs"] or len({job_key(j) for j in payload["jobs"]}) != len(payload["jobs"]):
        raise ValueError("registry must contain nonempty unique jobs")
    return payload


def run_job(job, source_manifest):
    start = time.perf_counter()
    comparison = None
    try:
        optimizer = JointOptimizer(OptimizerConfig(**job["config"]), job["bridge"], job["kappa"])
        result = optimizer.solve_mode(job["budget"], job["context"], job["form"], job["mode"])
        if job["mode"] != "fixed_p_q_baseline":
            baseline = optimizer.solve_mode(job["budget"], job["context"], job["form"], "fixed_p_q_baseline")
            comparison = {"baseline": baseline, "initial_candidate_loss": result["mean_loss"],
                          "retry": None, "absolute_tolerance": 1e-7}
            if result["mean_loss"] > baseline["mean_loss"] + 1e-7:
                config = {**job["config"], "starts": max(6, 2 * job["config"]["starts"])}
                expanded = JointOptimizer(OptimizerConfig(**config), job["bridge"], job["kappa"])
                retry = expanded.solve_mode(job["budget"], job["context"], job["form"], job["mode"])
                comparison["retry"] = {k: retry[k] for k in ("mean_loss", "optimizer_config", "attempts")}
                if retry["mean_loss"] > baseline["mean_loss"] + 1e-7:
                    raise RuntimeError("solver insufficiency: candidate remains worse than nested fixed-p/Q baseline")
                result = retry
            result["baseline_comparison"] = comparison
        status, error = "PASS", None
    except (ValueError, RuntimeError) as exc:
        status, result, error = "FAIL", None, str(exc)
    return {"status": status, "job": job, "result": result, "error": error,
            "baseline_comparison": comparison,
            "elapsed_seconds": time.perf_counter() - start, "source_manifest": source_manifest}


def task_exception_record(job, manifest, exc, execution_state):
    return {"status":"FAIL", "job":job, "result":None, "error":f"{type(exc).__name__}: {exc}",
            "exception_type":type(exc).__name__, "traceback":traceback.format_exc(),
            "execution_state":execution_state, "elapsed_seconds":None, "source_manifest":manifest,
            "scope":"execution failure evidence, not a converged scientific result"}


def smoke(output):
    model = Q3ModelContract()
    manifest = model.source_manifest()
    base = {"bridge": "ratio", "kappa": 1., "config": asdict(OptimizerConfig()), "scenario": "preflight"}
    jobs = [{**base, "budget": c, "context": context, "form": form, "mode": mode}
            for c, context, form in ((1e19, 2048, "exponential"), (1e22, 32768, "power"),
                                     (1e24, 131072, "log_asymptotic"))
            for mode in ("joint_reduced_hull", "fixed_p_q_baseline")]
    jobs += [{**base, "budget": 1e22, "context": 4096, "form": "exponential", "mode": mode}
             for mode in ("fixed_p", "fixed_q")]
    jobs += [{**base, "bridge": "additive", "budget": 1e22, "context": 8192,
              "form": "log_asymptotic", "mode": "joint_reduced_hull",
              "config": {**base["config"], "normalize_cost_endpoint": True}}]
    records = [run_job(job, manifest) for job in jobs]
    checks = {"all_jobs_converged": all(r["status"] == "PASS" for r in records)}
    if checks["all_jobs_converged"]:
        results = [r["result"] for r in records]
        checks["three_budget_orders"] = {r["budget_flops"] for r in results} == {1e19, 1e22, 1e24}
        checks["all_budget_feasible"] = all(r["budget_margin_flops"] >= -1e-10*r["budget_flops"] for r in results)
        checks["all_training_support_certified"] = all(r["support_certificate"]["inside_convex_hull"] for r in results)
        checks["joint_not_worse_than_included_baseline"] = all(
            results[i]["mean_loss"] <= results[i+1]["mean_loss"] + 1e-7 for i in (0, 2, 4))
        repeat = run_job(jobs[0], manifest)["result"]
        checks["seed_repeatable"] = all(np.allclose(results[0][k], repeat[k], rtol=1e-10, atol=1e-12)
                                        for k in ("N_params", "D_tokens", "Q", "p", "mean_loss"))
        costs = results[-1]["costs"]
        expected = model._g(1., "exponential") - model._g(model.q0, "exponential")
        actual = costs["quality_cost_scale"] * (model._g(1., "log_asymptotic") - model._g(model.q0, "log_asymptotic"))
        checks["normalized_cost_endpoint_matches"] = bool(np.isclose(expected, actual))
    try:
        JointOptimizer(OptimizerConfig(maxiter=1)).solve_mode(1e19, 2048, "exponential")
        checks["nonconvergence_is_rejected"] = False
    except RuntimeError:
        checks["nonconvergence_is_rejected"] = True
    status = "PASS" if all(checks.values()) else "FAIL"
    write_json(output, {"status": status, "scope": "preflight only, not the full experiment matrix",
                       "checks": checks, "records": records, "source_manifest": manifest,
                       "environment": {"python": sys.version, "numpy": np.__version__,
                                       "scipy": scipy.__version__, "platform": platform.platform()},
                       "claims_not_established": ["global optimality", "empirical bridge calibration",
                           "B1 outperformance", "stable structural transition", "formal uncertainty"]})
    print(json.dumps({"status": status, "checks": checks, "output": str(output)}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "register", "run"))
    parser.add_argument("--batch", choices=("main", "bridge", "search", "cost", "negative"), default="main")
    parser.add_argument("--workers", type=int, default=1, help="independent jobs; does not change registered mathematics")
    parser.add_argument("--output", default="results/q3_experiment_preflight.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budgets", type=float, nargs="+", help="optional refined FLOPs budget grid")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--registry", help="frozen registry name, defaults to batch; use a new name for refinement")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.budgets and (not np.isfinite(args.budgets).all() or min(args.budgets) <= 0 or len(set(args.budgets)) != len(args.budgets)):
        raise ValueError("budgets must be finite, positive and unique")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("outputs must remain within PROJECT_ROOT")
    if args.action == "smoke":
        smoke(output)
        return
    manifest = Q3ModelContract().source_manifest()
    registry_name = args.registry or args.batch
    if not re.fullmatch(r"[A-Za-z0-9_-]+", registry_name):
        raise ValueError("invalid registry name")
    registry_path = ROOT / f"results/registered_{registry_name}.json"
    if args.action == "register":
        jobs = list(registered_jobs(args.batch, sorted(args.budgets) if args.budgets else None))
        payload = registry_payload(args.batch, jobs, manifest)
        if registry_path.exists() and json.loads(registry_path.read_text()) != payload:
            raise RuntimeError("registry is immutable; archive old version or choose a new --registry name")
        write_json(registry_path, payload)
        print(json.dumps({"batch": args.batch, "count": len(jobs)}))
        return
    if args.budgets:
        raise ValueError("run reads only frozen registry; register refined budgets under a new name first")
    registry = load_registry(registry_path, manifest)
    jobs = registry["jobs"]
    gate = json.loads((ROOT / "qa/Q3正式实验准入.json").read_text(encoding="utf-8"))
    if gate.get("M1") != "PASS" or gate.get("P1") != "PASS" or gate.get("source_manifest") != manifest:
        raise RuntimeError("independent M1/P1 approval is missing or stale")
    failures, pending, records = [], [], {}
    for index, job in enumerate(jobs[:args.limit] if args.limit else jobs):
        identity = job_id(job)
        path = ROOT / f"results/formal/{registry['batch']}/{identity}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("source_manifest") != manifest:
                raise RuntimeError(f"stale batch output: {path}")
            if job_key(old["job"]) != job_key(job):
                raise RuntimeError(f"result configuration mismatch: {path}")
            record = old
            if old["status"] == "FAIL" and args.retry_failed:
                pending.append((job, path, old))
                continue
        else:
            pending.append((job, path, None))
            continue
        records[job_key(job)] = record
        if record["status"] != "PASS":
            failures.append(str(path))
        print(f"RESUME {index + 1}/{len(jobs)} {record['status']} {identity}", flush=True)

    def save_finished(record, path, old):
        if old is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            history = path.parent / "history" / path.stem / f"{digest}.json"
            history.parent.mkdir(parents=True, exist_ok=True)
            if history.exists() and history.read_bytes() != path.read_bytes():
                raise RuntimeError("immutable failure archive collision")
            if not history.exists():
                history.write_bytes(path.read_bytes())
            record["previous_failure_record"] = {"path": str(history.relative_to(ROOT)), "sha256": digest}
        write_json(path, record)
        records[job_key(record["job"])] = record
        if record["status"] != "PASS":
            failures.append(str(path))
        print(f"{len(records)}/{len(jobs)} {record['status']} {path.stem}", flush=True)

    if args.workers == 1:
        for job, path, old in pending:
            try:
                record = run_job(job, manifest)
            except Exception as exc:
                record = task_exception_record(job, manifest, exc, "started_failed")
            save_finished(record, path, old)
    else:
        unfinished = {job_key(job):(job,path,old) for job,path,old in pending}
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                for job, path, old in pending:
                    try:
                        futures[pool.submit(run_job, job, manifest)] = (job, path, old)
                    except Exception as exc:
                        save_finished(task_exception_record(job, manifest, exc, "not_started_submission_failed"),path,old)
                        unfinished.pop(job_key(job),None)
                for future in as_completed(futures):
                    job, path, old = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = task_exception_record(job, manifest, exc, "worker_failed_or_not_started")
                    save_finished(record,path,old)
                    unfinished.pop(job_key(job),None)
        except Exception as exc:
            for job,path,old in list(unfinished.values()):
                save_finished(task_exception_record(job,manifest,exc,"executor_failed_execution_unknown"),path,old)
    nested_violations = []
    for record in records.values():
        if record["status"] != "PASS" or record["job"]["mode"] != "joint_reduced_hull":
            continue
        for mode in ("fixed_p", "fixed_q", "fixed_p_q_baseline"):
            other = records.get(job_key({**record["job"], "mode": mode}))
            if other and other["status"] == "PASS" and record["result"]["mean_loss"] > other["result"]["mean_loss"] + 1e-7:
                nested_violations.append({"joint_id": job_id(record["job"]), "better_mode": mode,
                                          "loss_gap": record["result"]["mean_loss"]-other["result"]["mean_loss"]})
    write_json(ROOT / f"results/formal/{registry['batch']}/batch_status.json", {
        "registry": str(registry_path.relative_to(ROOT)), "registry_sha256": registry["registry_sha256"],
        "planned": len(jobs), "attempted": sum(r.get("execution_state", "started") in ("started","started_failed") for r in records.values()),
        "task_outcomes_recorded": len(records),
        "failures": failures, "nested_comparison_violations": nested_violations,
        "workers": args.workers, "source_manifest": manifest,
        "status": "FAIL" if failures or nested_violations else ("PARTIAL" if args.limit and args.limit < len(jobs) else "PASS")})
    write_json(ROOT / f"results/batch_status/{registry_name}.json", json.loads(
        (ROOT / f"results/formal/{registry['batch']}/batch_status.json").read_text()))
    if failures or nested_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
