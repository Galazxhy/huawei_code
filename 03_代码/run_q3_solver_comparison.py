"""Fixed-problem, equal model-evaluation allowance solver comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import itertools
import json
import time

from q3_model_contract import ROOT, Q3ModelContract
from q3_optimizer import JointOptimizer, OptimizerConfig
from run_q3_experiments import job_id, job_key, load_registry, registry_payload, write_json, task_exception_record


def solve(job, manifest):
    start = time.perf_counter()
    try:
        result = JointOptimizer(OptimizerConfig(**job["config"]), job["bridge"], job["kappa"]).solve_mode(
            job["budget"], job["context"], job["form"], job["mode"])
        status, error = "PASS", None
    except (ValueError, RuntimeError) as exc:
        result, status, error = None, "FAIL", str(exc)
    return {"job": job, "source_manifest": manifest, "result": result, "status": status,
            "error": error, "elapsed_seconds": time.perf_counter()-start,
            "scope": "same reduced hull and starts; maximum 3000 distinct response calls per start"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    model = Q3ModelContract()
    manifest = model.source_manifest()
    registry_path = ROOT / "results/solver_comparison_registry.json"
    if args.action == "register":
        jobs = []
        for budget, context, form, solver in itertools.product(
                (1e19, 1e20, 3e20, 1e22, 1e24), (2048, 131072),
                ("exponential", "power", "log_asymptotic"), ("SLSQP", "COBYQA")):
            config = asdict(OptimizerConfig(solver=solver, maxiter=4000, max_model_evaluations_per_start=3000))
            jobs.append({"scenario": "fixed_problem_solver_comparison", "budget": budget, "context": context,
                         "form": form, "bridge": "ratio", "kappa": 1., "mode": "joint_reduced_hull", "config": config})
        payload = registry_payload("solver_comparison", jobs, manifest)
        if registry_path.exists() and json.loads(registry_path.read_text()) != payload:
            raise RuntimeError("solver registry is immutable")
        write_json(registry_path, payload)
        print(json.dumps({"registered": len(jobs), "per_start_model_evaluation_cap": 3000}))
        return
    registry = load_registry(registry_path, manifest)
    gate = json.loads((ROOT / "qa/Q3正式实验准入.json").read_text())
    if gate.get("M1") != "PASS" or gate.get("P1") != "PASS" or gate.get("source_manifest") != manifest:
        raise RuntimeError("independent V002 admission missing or stale")
    directory = ROOT / "results/solver_comparison"
    directory.mkdir(exist_ok=True)
    pending, records = [], []
    for job in registry["jobs"]:
        path = directory / (job_id(job)+".json")
        if path.exists():
            old = json.loads(path.read_text())
            if old["source_manifest"] != manifest or old["job"] != job:
                raise RuntimeError("stale solver comparison result")
            records.append(old)
        else:
            pending.append((job, path))
    unfinished = {job_key(job):(job,path) for job,path in pending}
    def save(record,path):
        write_json(path,record)
        records.append(record)
        unfinished.pop(job_key(record["job"]),None)
        print(f"{len(records)}/{len(registry['jobs'])} {record['status']}",flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for job,path in pending:
                try:
                    futures[pool.submit(solve,job,manifest)] = (job,path)
                except Exception as exc:
                    save(task_exception_record(job,manifest,exc,"not_started_submission_failed"),path)
            for future in as_completed(futures):
                job,path = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = task_exception_record(job,manifest,exc,"worker_failed_or_not_started")
                save(record,path)
    except Exception as exc:
        for job,path in list(unfinished.values()):
            save(task_exception_record(job,manifest,exc,"executor_failed_execution_unknown"),path)
    counts = {solver: {"planned": 0, "converged_admissible": 0} for solver in ("SLSQP", "COBYQA")}
    pairs = {}
    for record in records:
        solver = record["job"]["config"]["solver"]
        counts[solver]["planned"] += 1
        counts[solver]["converged_admissible"] += record["status"] == "PASS"
        key = json.dumps([record["job"][x] for x in ("budget", "context", "form")])
        pairs.setdefault(key, {})[solver] = record
    differences = []
    for key, pair in pairs.items():
        if all(pair[s]["status"] == "PASS" for s in counts):
            results = {s: pair[s]["result"] for s in counts}
            identical_starts = all(results["SLSQP"]["attempts"][i]["initial_vector"] ==
                                   results["COBYQA"]["attempts"][i]["initial_vector"] for i in range(3))
            differences.append({"configuration": json.loads(key), "identical_starts": identical_starts,
                                "loss_SLSQP_minus_COBYQA": results["SLSQP"]["mean_loss"]-results["COBYQA"]["mean_loss"],
                                "model_calls": {s: results[s]["total_model_evaluations"] for s in counts},
                                "wall_seconds": {s: pair[s]["elapsed_seconds"] for s in counts}})
            if not identical_starts:
                raise RuntimeError("solver starts differ")
    write_json(directory / "comparison_summary.json", {
        "status": "COMPLETE" if len(records) == len(registry["jobs"]) else "INCOMPLETE",
        "counts": counts, "paired_comparisons": differences,
        "failures": [job_id(r["job"]) for r in records if r["status"] != "PASS"],
        "all_cases_retained": True, "failure_semantics": "nonconvergence under fixed allowance; not missing",
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "source_manifest": manifest})


if __name__ == "__main__":
    main()
