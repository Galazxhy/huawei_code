"""Bounded analysis of registered Q3 candidates; no optimization or approval.

Authoritative JSON preserves floats. Read exported CSV with round_trip.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import uuid

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
V001 = ROOT.parents[1] / "V001/PZ_Q3"
QA = ROOT / "qa/分析自检"
MODES = ("fixed_p_q_baseline", "fixed_p", "fixed_q", "joint_reduced_hull")
NESTED = {(MODES[0], m) for m in MODES[1:]} | {(m, MODES[3]) for m in MODES[1:3]}
SCENARIOS = [(b, k) for b in ("ratio", "additive") for k in (.5, 1., 1.5)]
LOSS_ATOL = 1e-9
NESTED_ATOL = 1e-7


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def under(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path outside allowed root: {path}")
    return path


def write_json(path, value):
    path = Path(path).resolve()
    if not any(path.is_relative_to(p.resolve()) for p in (ROOT / "results/analysis", QA)):
        raise ValueError(f"output outside permitted analysis directories: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def table(target, name, rows):
    write_json(target / f"{name}.json", rows)
    if rows:
        flat = [{k: canonical(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()} for row in rows]
        pd.DataFrame(flat).to_csv(target / f"{name}.csv", index=False, float_format="%.17g")


@contextmanager
def round_trip_csv():
    original = pd.read_csv

    def read(*args, **kwargs):
        kwargs.setdefault("float_precision", "round_trip")
        return original(*args, **kwargs)

    pd.read_csv = read
    try:
        yield
    finally:
        pd.read_csv = original


class Inputs:
    def __init__(self, source):
        self.source, self.hashes = source, {}

    def track(self, path):
        path = under(path, self.source)
        value, key = digest(path), str(path)
        if key in self.hashes and self.hashes[key] != value:
            raise ValueError(f"input changed during analysis: {path}")
        self.hashes[key] = value
        return path

    def read(self, path):
        path = self.track(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        self.track(path)
        return value

    def changed(self):
        return [p for p, h in self.hashes.items() if not Path(p).is_file() or digest(p) != h]


def load_entries(names, data_root, fixture, ev, inputs):
    entries, issues, registries, catalog = [], [], [], {}
    owners = defaultdict(list)
    for path in sorted(data_root.glob("registered_*.json")):
        name = path.stem.removeprefix("registered_")
        try:
            raw = inputs.read(path)
            payload = ev.runner.load_registry(path, ev.manifest)
            if payload != raw or not re.fullmatch(r"[A-Za-z0-9_-]+", payload["batch"]):
                raise ValueError("changed registry or invalid batch")
            catalog[name] = payload
            for job in payload["jobs"]:
                owners[(payload["batch"], ev.runner.job_id(job))].append(name)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            issues.append({"kind": "invalid_registry", "registry": name, "error": str(exc),
                           "severity": "error" if name in names else "information",
                           "used_for_results": name in names})
    for name in names:
        if name not in catalog:
            if not (data_root / f"registered_{name}.json").exists():
                issues.append({"kind": "missing_registry", "registry": name})
            continue
        registry = catalog[name]
        batch = registry["batch"]
        registries.append({"name": name, "batch": batch, "jobs": len(registry["jobs"]),
                           "registry_sha256": registry["registry_sha256"]})
        folder = data_root / ("records" if fixture else "formal") / batch
        for job in registry["jobs"]:
            jid = ev.runner.job_id(job)
            path = folder / f"{jid}.json"
            entry = {"entry_id": f"{name}:{jid}", "registry": name, "batch": batch, "job_id": jid,
                     "job": job, "state": "missing", "source_file": str(path)}
            entries.append(entry)
            if not path.is_file():
                entry["error"] = "registered result absent"
                continue
            try:
                record = inputs.read(path)
                entry["input_status"] = record.get("status")
                if not fixture and ("fixture_scope" in record or record.get("fixture_only")):
                    raise ValueError("fixture record cannot be used as formal evidence")
                if record["source_manifest"] != ev.manifest:
                    raise ValueError("result source_manifest mismatch")
                if ev.runner.job_key(record["job"]) != ev.runner.job_key(job):
                    raise ValueError("result job identity mismatch")
                if job["mode"] not in MODES:
                    raise ValueError("unsupported mode")
                if record["status"] != "PASS":
                    entry.update(state="failed", error=record.get("error") or "upstream record not accepted")
                    continue
                result = record["result"]
                value = ev.validate(job, result)
                entry.update(state="available", result=result, evaluated=value, conditions=ev.conditions(job),
                             protocol_conditions=ev.conditions(job, True))
                comparison = result.get("baseline_comparison") or record.get("baseline_comparison")
                if comparison and comparison.get("baseline"):
                    try:
                        baseline = comparison["baseline"]
                        entry["embedded_baseline"] = {"result": baseline,
                            "evaluated": ev.validate({**job, "mode": MODES[0]}, baseline)}
                    except (ValueError, KeyError, TypeError) as exc:
                        issues.append({"kind": "invalid_embedded_baseline", "entry_id": entry["entry_id"], "error": str(exc)})
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                entry.update(state="invalid", error=str(exc))
        specific = data_root / "batch_status" / f"{name}.json"
        status_path = specific if specific.is_file() else folder / "batch_status.json"
        if not status_path.is_file():
            issues.append({"kind": "missing_batch_status", "registry": name})
        else:
            try:
                status = inputs.read(status_path)
                if status["source_manifest"] != ev.manifest:
                    raise ValueError("batch status manifest mismatch")
                if status["registry_sha256"] != registry["registry_sha256"]:
                    other = [n for n, r in catalog.items() if r["batch"] == batch and r["registry_sha256"] == status["registry_sha256"]]
                    if not specific.is_file() and other:
                        issues.append({"kind": "shared_batch_status_describes_other_registry", "registry": name,
                                       "severity": "information", "status_owners": other,
                                       "detail": "This registry is checked from its individual records."})
                        continue
                    raise ValueError("batch status registry hash mismatch")
                if status["planned"] != len(registry["jobs"]):
                    raise ValueError("batch planned count mismatch")
                if status["status"] != "PASS" or status.get("failures") or status.get("nested_comparison_violations"):
                    issues.append({"kind": "upstream_batch_incomplete_or_failed", "registry": name, "detail": status})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append({"kind": "invalid_batch_status", "registry": name, "error": str(exc)})
    selected_batches = {r["batch"] for r in registries}
    for batch in sorted(selected_batches):
        for path in sorted((data_root / ("records" if fixture else "formal") / batch).glob("*.json")):
            if path.name == "batch_status.json":
                continue
            known = owners[(batch, path.stem)]
            if not any(name in names for name in known):
                inputs.track(path)
                issues.append({"kind": "registered_result_outside_request" if known else "unregistered_result_in_batch",
                               "severity": "information" if known else "error", "registered_owners": known, "path": str(path)})
    return entries, issues, registries


def evidence_rows(entries, ev):
    rows, starts = [], []
    for e in entries:
        if e["state"] != "available":
            continue
        j, r, value = e["job"], e["result"], e["evaluated"]
        p = np.asarray(value["p"])
        scales = np.asarray(ev.base.observed_scale_anchors)
        distances = np.linalg.norm(np.log10([value["N_params"], value["D_tokens"]])-np.log10(scales), axis=1)
        near = int(np.argmin(distances))
        accepted = [a["objective"] for a in r["attempts"] if a.get("accepted")]
        e["start_range"] = float(np.ptp(accepted))
        rows.append({"entry_id": e["entry_id"], "registry": e["registry"], "scenario": j["scenario"],
            "mode": j["mode"], "bridge": j["bridge"], "kappa": j["kappa"], **e["conditions"],
            "N": value["N_params"], "D": value["D_tokens"], "Q": value["Q"], "p": value["p"],
            "mean_loss": value["mean_loss"], "loss_by_domain": value["loss_by_domain"], "costs": value["costs"],
            "budget_margin": value["budget_margin_flops"], "nearest_observed_scale": scales[near].tolist(),
            "nearest_scale_log10_distance": float(distances[near]),
            "nearest_training_mixture_l1": float(np.abs(ev.base.support_points-p).sum(axis=1).min()),
            "evidence_geometry": "A4 full hull and registered reduced anchor hull certificates",
            "evidence_scale": "observed scale" if value["scale_supported_by_A_mixture"] else "conditional extrapolation",
            "evidence_quality": "null effect control" if j["kappa"] == 0 else "uncalibrated B7 quality transfer",
            "evidence_cost": "specified prices on assumed common quality scale", "distances_are_error_bounds": False,
            "registered_starts": j["config"]["starts"], "actual_starts": len(r["attempts"]), "accepted_starts": len(accepted),
            "accepted_start_min_loss": min(accepted), "accepted_start_max_loss": max(accepted),
            "accepted_start_loss_range": e["start_range"], "numerical_spread_available": len(accepted) >= 2,
            "feasible_perturbation_check": r.get("feasible_perturbation_pass"),
            "full_hull_certificate": value["support_certificate"], "anchor_hull_certificate": ev.hull(j, p)})
        starts.extend({"entry_id": e["entry_id"], **a} for a in r["attempts"])
    return rows, starts

class Evaluator:
    """Load the selected version's read-only interfaces; never invoke its solver."""
    def __init__(self, source, inputs):
        for path in sorted((source / "02_代码与实验").glob("*.py")):
            inputs.track(path)
        sys.path.insert(0, str(source / "02_代码与实验"))
        self.contract = importlib.import_module("q3_model_contract")
        self.optimizer = importlib.import_module("q3_optimizer")
        self.runner = importlib.import_module("run_q3_experiments")
        if Path(self.contract.ROOT).resolve() != source:
            raise ValueError("module root mismatch: each source version requires its own process")
        self.manifest = self.contract.Q3ModelContract.source_manifest(None)
        for relative, expected in self.manifest.items():
            if digest(inputs.track(source.parent / relative)) != expected:
                raise ValueError(f"manifest mismatch: {relative}")
        with round_trip_csv():
            self.base = self.contract.Q3ModelContract()
        if self.base.source_manifest() != self.manifest:
            raise ValueError("inputs changed during model initialization")
        self.models, self.hulls = {}, {}

    @functools.lru_cache(maxsize=None)
    def full_certificate(self, p):
        return self.base.support_certificate(np.asarray(p))

    def configuration(self, job):
        return asdict(self.optimizer.OptimizerConfig(**job["config"]))

    def model(self, job):
        c = self.configuration(job)
        key = canonical([job["bridge"], job["kappa"], c["n_bounds"], c["d_bounds"],
                         c["q0_override"], c["normalize_cost_endpoint"], job["form"]])
        if key not in self.models:
            model = copy.copy(self.base)
            model.q0 = self.base.q0 if c["q0_override"] is None else float(c["q0_override"])
            if not np.isfinite(model.q0) or not 0 < model.q0 <= 1:
                raise ValueError("invalid Q0")
            if job["bridge"] not in ("ratio", "additive") or not np.isfinite(job["kappa"]) or job["kappa"] < 0:
                raise ValueError("invalid bridge scenario")
            model.config = replace(self.base.config, quality_bridge_mode=job["bridge"], quality_bridge_kappa=job["kappa"],
                n_bounds=tuple(c["n_bounds"]), d_bounds=tuple(c["d_bounds"]), q0_override=c["q0_override"], quality_cost_scale=1.)
            if c["normalize_cost_endpoint"]:
                increment = model._g(1., job["form"]) - model._g(model.q0, job["form"])
                if increment <= 0:
                    raise ValueError("nonpositive normalization endpoint")
                scale = (model._g(1., "exponential") - model._g(model.q0, "exponential")) / increment
                model.config = replace(model.config, quality_cost_scale=scale)
            model.support_certificate = lambda p: self.full_certificate(tuple(float(x) for x in p))
            self.models[key] = model
        return self.models[key]

    @functools.lru_cache(maxsize=None)
    def anchors(self, count):
        return self.optimizer.farthest_anchors(self.base.support_points, count)

    def hull(self, job, p):
        count = self.configuration(job)["anchor_count"]
        p = self.base._validate_probability(p)
        key = (count, tuple(float(x) for x in p))
        if key not in self.hulls:
            anchors = self.anchors(count)
            matrix = np.vstack([anchors.T, np.ones(len(anchors))])
            rhs = np.r_[p, 1.]
            scale = np.maximum(np.max(np.abs(matrix), axis=1), 1e-12)
            attempts, inside, error = [], False, None
            # Near a hull face, presolve may falsely report infeasibility. Retry
            # an equivalent scaled LP, then verify the certificate in original units.
            for label, mat, vec, extra in (("direct", matrix, rhs, {}),
                    ("scaled_without_presolve", matrix/scale[:, None], rhs/scale, {"presolve": False})):
                error = None
                fit = linprog(np.zeros(len(anchors)), A_eq=mat, b_eq=vec, bounds=(0., None), method="highs",
                    options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9, **extra})
                if fit.success and np.isfinite(fit.x).all() and np.maximum(fit.x, 0).sum() > 0:
                    weights = np.maximum(fit.x, 0)
                    weights /= weights.sum()
                    error = float(np.max(np.abs(anchors.T @ weights - p)))
                    inside = bool(error <= self.base.config.support_tolerance)
                attempts.append({"method": label, "solver_success": bool(fit.success),
                                 "original_unit_reconstruction_error": error, "certificate_accepted": inside,
                                 "message": str(fit.message)})
                if inside:
                    break
            self.hulls[key] = {"inside": inside, "max_reconstruction_error": error,
                               "message": str(fit.message), "lp_attempts": attempts}
        return self.hulls[key]

    def conditions(self, job, protocol=False):
        c, model = self.configuration(job), self.model(job)
        result = {"budget": job["budget"], "context": job["context"], "form": job["form"], "Q0": model.q0,
            "n_bounds": c["n_bounds"], "d_bounds": c["d_bounds"], "anchor_count": c["anchor_count"],
            "anchor_sha256": hashlib.sha256(self.anchors(c["anchor_count"]).tobytes()).hexdigest(),
            "normalize_cost_endpoint": c["normalize_cost_endpoint"], "quality_cost_scale": model.config.quality_cost_scale,
            "eta_attention": model.config.eta_attention}
        if protocol:
            result["search_protocol"] = {k: v for k, v in c.items() if k not in
                ("n_bounds", "d_bounds", "anchor_count", "q0_override", "normalize_cost_endpoint")}
        return result

    def evaluate(self, job, result, mode_guard=True):
        model, p = self.model(job), np.asarray(result["p"], dtype=float)
        if not self.hull(job, p)["inside"]:
            raise ValueError("recipe outside registered reduced anchor hull")
        if mode_guard:
            if job["mode"] in MODES[:2] and not np.allclose(p, self.base.support_points.mean(axis=0), atol=2e-8, rtol=0):
                raise ValueError("fixed-p mode changed reference recipe")
            if job["mode"] in (MODES[0], MODES[2]) and abs(result["Q"]-model.q0) > 2e-8:
                raise ValueError("fixed-Q mode changed Q0")
        value = model.evaluate(result["N_params"], result["D_tokens"], p, result["Q"], job["context"],
                               job["form"], budget_flops=job["budget"])
        if min(value["loss_by_domain"].values()) < 1e-6:
            raise ValueError("candidate exploits positivity boundary (<1e-6)")
        return value

    def validate(self, job, result):
        value = self.evaluate(job, result)
        if result.get("solver_success") is not True or result.get("optimization_mode") != job["mode"]:
            raise ValueError("solver convergence or decision mode mismatch")
        if set(result["loss_by_domain"]) != set(self.base.loss_domains):
            raise ValueError("loss domain set mismatch")
        for key in ("Q0", "mean_loss"):
            if not np.isclose(result[key], value[key], rtol=0, atol=LOSS_ATOL):
                raise ValueError(f"recomputed {key} mismatch")
        for key, expected in (("budget_flops", job["budget"]), ("L_ctx", job["context"]),
                              ("quality_bridge_mode", job["bridge"]), ("quality_bridge_kappa", job["kappa"])):
            if result.get(key) != expected:
                raise ValueError(f"result metadata mismatch: {key}")
        if not np.allclose([result["loss_by_domain"][d] for d in self.base.loss_domains],
                          [value["loss_by_domain"][d] for d in self.base.loss_domains], rtol=0, atol=LOSS_ATOL):
            raise ValueError("recomputed domain Loss mismatch")
        for key in ("C_train_flops", "C_attn_flops", "C_quality_flops", "C_total_flops"):
            if not np.isclose(result["costs"][key], value["costs"][key], rtol=1e-10, atol=1.):
                raise ValueError(f"recomputed cost mismatch: {key}")
        actual = asdict(self.optimizer.OptimizerConfig(**result["optimizer_config"]))
        for key, registered in self.configuration(job).items():
            if (key == "starts" and actual[key] < registered) or (key != "starts" and canonical(actual[key]) != canonical(registered)):
                raise ValueError(f"actual optimizer configuration mismatch: {key}")
        attempts = result["attempts"]
        if len(attempts) != actual["starts"] or len({a["start"] for a in attempts}) != len(attempts):
            raise ValueError("missing or duplicate start records")
        accepted = [a for a in attempts if a.get("accepted")]
        if not accepted or any(not a.get("success") or not np.isfinite(a["objective"]) for a in accepted):
            raise ValueError("invalid accepted starts")
        if not np.isclose(min(a["objective"] for a in accepted), result["mean_loss"], rtol=0, atol=LOSS_ATOL):
            raise ValueError("selected result is not the best accepted start")
        if result["best_start"] not in {a["start"] for a in accepted}:
            raise ValueError("selected start was not accepted")
        return value


def scientific_comparison(a, b):
    fields = {"N_params": (1e-10, 1e-6), "D_tokens": (1e-10, 1e-6), "Q": (0., 1e-10),
              "Q0": (0., 1e-12), "p": (0., 2e-8), "mean_loss": (0., LOSS_ATOL),
              "loss_by_domain": (0., LOSS_ATOL), "costs": (1e-10, 1.)}
    checks = {}
    for key, (rtol, atol) in fields.items():
        left, right = a[key], b[key]
        if key == "loss_by_domain":
            if set(left) != set(right):
                checks[key] = {"equal": False, "error": "domain set mismatch"}
                continue
            left, right = [left[d] for d in sorted(left)], [right[d] for d in sorted(right)]
        if key == "costs":
            keys = ("C_train_flops", "C_attn_flops", "C_quality_flops", "C_total_flops")
            left, right = [left[k] for k in keys], [right[k] for k in keys]
        left, right = np.asarray(left), np.asarray(right)
        checks[key] = {"equal": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
                       "rtol": rtol, "atol": atol, "max_absolute_difference": float(np.max(np.abs(left-right)))}
    def start_science(result):
        attempts = result["attempts"]
        accepted = sorted(canonical({"objective": float(attempt["objective"]),
                                     "p": [float(v) for v in attempt["p"]]})
                          for attempt in attempts if attempt.get("accepted"))
        return {"attempt_count": len(attempts), "accepted_candidates": accepted}

    left_starts, right_starts = start_science(a), start_science(b)
    checks["accepted_start_evidence"] = {
        "equal": left_starts == right_starts,
        "comparison": "exact unordered multiset of accepted objective/p plus total attempt count",
        "reference_attempts": left_starts["attempt_count"],
        "candidate_attempts": right_starts["attempt_count"],
        "reference_accepted": len(left_starts["accepted_candidates"]),
        "candidate_accepted": len(right_starts["accepted_candidates"]),
        "reference_sha256": hashlib.sha256(canonical(left_starts).encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(canonical(right_starts).encode()).hexdigest()}
    return {"equal_within_tolerance": all(c["equal"] for c in checks.values()), "fields": checks,
            "ignored_fields": ["elapsed_seconds", "timing", "optimizer messages"],
            "scope": "selected solution and accepted-start statistics; each source retains its metadata"}


def duplicate_checks(entries, ev):
    groups = defaultdict(list)
    for e in entries:
        groups[ev.runner.job_key(e["job"])].append(e)
    checks = []
    for values in groups.values():
        for a, b in itertools.combinations(values, 2):
            row = {"reference_id": a["entry_id"], "candidate_id": b["entry_id"],
                   "reference_source": a["source_file"], "candidate_source": b["source_file"],
                   "reference_state": a["state"], "candidate_state": b["state"]}
            if a["state"] == b["state"] == "available":
                row.update(scientific_comparison(a["result"], b["result"]))
                row["state"] = "equivalent" if row["equal_within_tolerance"] else "scientific_mismatch"
            else:
                row["state"] = "incomplete_sources"
            checks.append(row)
    return checks


def pairing(entries, ev):
    groups, repeated = defaultdict(dict), defaultdict(list)
    for e in entries:
        key = canonical([e["registry"], {k: v for k, v in e["job"].items() if k != "mode"}])
        groups[key][e["job"]["mode"]] = e
        if e["state"] == "available":
            repeated[canonical([e["conditions"], e["job"]["bridge"], e["job"]["kappa"], e["job"]["mode"]])].append(e)
    sensitivity = []
    for values in repeated.values():
        span = max(e["result"]["mean_loss"] for e in values)-min(e["result"]["mean_loss"] for e in values)
        for e in values:
            e["same_problem_repeat_range"] = span
        for a, b in itertools.combinations(values, 2):
            if a["protocol_conditions"] == b["protocol_conditions"]:
                continue
            gap = a["result"]["mean_loss"]-b["result"]["mean_loss"]
            sensitivity.append({"reference_id": a["entry_id"], "candidate_id": b["entry_id"],
                "reference_protocol": a["protocol_conditions"]["search_protocol"],
                "candidate_protocol": b["protocol_conditions"]["search_protocol"],
                "signed_loss_gain": gap, "absolute_loss_gap": abs(gap), "conditions": a["conditions"],
                "bridge": a["job"]["bridge"], "kappa": a["job"]["kappa"], "mode": a["job"]["mode"]})
    pairs, domains, four = [], [], []
    for key, modes in groups.items():
        gid, anchor = identity(key), next(iter(modes.values()))
        common = {"group_id": gid, "registry": anchor["registry"],
                  "job_without_mode": {k: v for k, v in anchor["job"].items() if k != "mode"}}
        for ref, candidate in itertools.combinations(MODES, 2):
            a, b = modes.get(ref), modes.get(candidate)
            row = {**common, "reference_mode": ref, "candidate_mode": candidate, "nested": (ref, candidate) in NESTED,
                   "reference_state": a["state"] if a else "not_registered", "candidate_state": b["state"] if b else "not_registered"}
            if not a or not b or a["state"] != "available" or b["state"] != "available":
                pairs.append({**row, "state": "incomplete"})
                continue
            gains = []
            for domain in ev.base.loss_domains:
                la, lb = a["result"]["loss_by_domain"][domain], b["result"]["loss_by_domain"][domain]
                gain = la-lb
                gains.append(gain)
                domains.append({"group_id": gid, "reference_id": a["entry_id"], "candidate_id": b["entry_id"],
                    "reference_mode": ref, "candidate_mode": candidate, "domain": domain,
                    "reference_loss": la, "candidate_loss": lb, "gain_abs": gain, "gain_percent": 100*gain/la,
                    "direction": "improved" if gain > LOSS_ATOL else "degraded" if gain < -LOSS_ATOL else "numerical_tie"})
            gain = a["result"]["mean_loss"]-b["result"]["mean_loss"]
            within = a["start_range"]+b["start_range"]
            repeats = a["same_problem_repeat_range"]+b["same_problem_repeat_range"]
            comparator = max(within, repeats)
            enough = all(sum(x.get("accepted", False) for x in e["result"]["attempts"]) >= 2 for e in (a, b))
            violation = bool(row["nested"] and gain < -NESTED_ATOL)
            pairs.append({**row, "state": "computed", "reference_id": a["entry_id"], "candidate_id": b["entry_id"],
                "mean_gain_abs": gain, "mean_gain_percent": 100*gain/a["result"]["mean_loss"],
                "worst_domain": ev.base.loss_domains[int(np.argmin(gains))], "minimum_signed_domain_gain": min(gains),
                "maximum_domain_degradation": max(0., -min(gains)), "improved_domains": sum(g > LOSS_ATOL for g in gains),
                "degraded_domains": sum(g < -LOSS_ATOL for g in gains), "tied_domains": sum(abs(g) <= LOSS_ATOL for g in gains),
                "nested_violation": violation, "publication_blocked": violation, "nested_absolute_tolerance": NESTED_ATOL,
                "sum_accepted_start_ranges": within, "sum_same_problem_repeat_ranges": repeats, "numerical_comparator": comparator,
                "numerical_comparator_sufficient_starts": enough, "gain_exceeds_numerical_comparator": bool(enough and gain > comparator+LOSS_ATOL),
                "gain_to_numerical_comparator": gain/comparator if comparator > 0 else None, "comparison_is_statistical_test": False})
        missing = [m for m in MODES if m not in modes or modes[m]["state"] != "available"]
        row = {**common, "state": "incomplete" if missing else "computed", "unavailable_modes": missing}
        if not missing:
            b, qp, pq, joint = [modes[m]["result"]["mean_loss"] for m in MODES]
            row.update(joint_gain=b-joint, quality_only_gain=b-qp, mixture_only_gain=b-pq,
                joint_minus_additive_gain=qp+pq-b-joint,
                interpretation="shared budget and N/D reoptimization; not identified direct Q-by-p interaction")
        four.append(row)
    return pairs, domains, four, sensitivity


def cross_evaluate(candidates, representative, scenarios, ev):
    rows, common = [], []
    for c in candidates:
        accepted, cells = True, []
        for bridge, kappa in scenarios:
            job = {**representative, "bridge": bridge, "kappa": kappa}
            row = {"candidate_id": c["candidate_id"], "bridge": bridge, "kappa": kappa}
            try:
                r, model = c["result"], ev.model(job)
                costs = model.costs(r["N_params"], r["D_tokens"], r["Q"], job["context"], job["form"])
                row.update(costs=costs, budget_margin=job["budget"]-costs["C_total_flops"])
                value = ev.evaluate(job, r, mode_guard=False)
                row.update(state="feasible", loss=value["mean_loss"], loss_by_domain=value["loss_by_domain"])
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                row.update(state="rejected", error=str(exc))
                accepted = False
            cells.append(row)
        rows.extend({**r, "common_feasible": accepted} for r in cells)
        if accepted:
            common.append(c)
    best = {}
    for scenario in scenarios:
        values = [r["loss"] for r in rows if r["common_feasible"] and (r["bridge"], r["kappa"]) == scenario]
        if values:
            best[scenario] = min(values)
    for r in rows:
        if r["common_feasible"]:
            optimum = best[(r["bridge"], r["kappa"])]
            r.update(candidate_set_best_loss=optimum, regret_loss=r["loss"]-optimum,
                     relative_regret=(r["loss"]-optimum)/abs(optimum))
    summaries = []
    for c in common:
        own = [r for r in rows if r["candidate_id"] == c["candidate_id"]]
        summaries.append({"candidate_id": c["candidate_id"], "includes_baseline": c["includes_baseline"],
                          "maximum_regret_loss": max(r["regret_loss"] for r in own),
                          "maximum_relative_regret": max(r["relative_regret"] for r in own)})
    if summaries:
        minimum = min(r["maximum_regret_loss"] for r in summaries)
        relative = min(r["maximum_relative_regret"] for r in summaries)
        for r in summaries:
            r.update(minimax_loss_candidate=abs(r["maximum_regret_loss"]-minimum) <= LOSS_ATOL,
                     minimax_relative_candidate=abs(r["maximum_relative_regret"]-relative) <= LOSS_ATOL)
    return rows, summaries, common


def ranking(candidates, representative, ev):
    if not candidates:
        return [], []
    baseline = next((c for c in candidates if c["includes_baseline"]), candidates[0])
    high = max(candidates, key=lambda c: (c["result"]["Q"], c["candidate_id"]))
    probes = {canonical([c["result"][k] for k in ("N_params", "D_tokens", "Q")]): c for c in (baseline, high)}
    recipes = {identity(c["result"]["p"]): c["result"]["p"] for c in candidates}
    p0 = ev.base.support_points.mean(axis=0).tolist()
    recipes[identity(p0)] = p0
    checks, values = [], []
    for probe in probes.values():
        point, evaluations, rejected = probe["result"], {}, []
        for pid, p in recipes.items():
            found = []
            for bridge, kappa in SCENARIOS:
                try:
                    v = ev.evaluate({**representative, "bridge": bridge, "kappa": kappa}, {**point, "p": p}, mode_guard=False)
                    base = ev.base.q2.predict(point["N_params"], point["D_tokens"], np.asarray(v["p"]))
                    final = np.array([v["loss_by_domain"][d] for d in ev.base.loss_domains])
                    found.append((base, final))
                except (ValueError, RuntimeError) as exc:
                    rejected.append({"recipe_id": pid, "bridge": bridge, "kappa": kappa, "error": str(exc)})
            if len(found) == len(SCENARIOS):
                evaluations[pid] = found
        for i, (bridge, kappa) in enumerate(SCENARIOS):
            model = ev.model({**representative, "bridge": bridge, "kappa": kappa})
            f = model._quality_loss(point["N_params"], point["D_tokens"], point["Q"])
            f0 = model._quality_loss(point["N_params"], point["D_tokens"], model.q0)
            multiplier, offset = ((f/f0)**kappa, 0.) if bridge == "ratio" else (1., kappa*(f-f0))
            row = {"probe_candidate_id": probe["candidate_id"], "N": point["N_params"], "D": point["D_tokens"], "Q": point["Q"],
                "bridge": bridge, "kappa": kappa, "multiplier": multiplier, "offset": offset, "recipes_planned": len(recipes),
                "common_feasible_recipes": len(evaluations), "rejected_recipe_scenarios": rejected, "tolerance": LOSS_ATOL}
            if len(evaluations) < 2:
                checks.append({**row, "state": "insufficient_common_recipes"})
                continue
            base = np.array([v[i][0] for v in evaluations.values()])
            final = np.array([v[i][1] for v in evaluations.values()])
            base, final = np.column_stack([base, base.mean(axis=1)]), np.column_stack([final, final.mean(axis=1)])
            a, b = np.triu_indices(len(base), 1)
            db, df = base[a]-base[b], final[a]-final[b]
            tol = LOSS_ATOL*np.maximum(1., np.maximum(np.abs(final[a]), np.abs(final[b])))
            reversals = int(np.count_nonzero(((db > LOSS_ATOL) & (df < -tol)) | ((db < -LOSS_ATOL) & (df > tol))))
            residual = float(np.max(np.abs(final-(base*multiplier+offset))))
            checks.append({**row, "state": "computed", "strict_order_reversals": reversals,
                "base_near_tie_comparisons": int(np.count_nonzero(abs(db) <= LOSS_ATOL)), "pair_metric_comparisons": int(db.size),
                "max_affine_identity_error": residual, "ordering_identity_holds": bool(multiplier > 0 and reversals == 0 and residual <= LOSS_ATOL)})
            for index, pid in enumerate(evaluations):
                values.append({"probe_candidate_id": probe["candidate_id"], "recipe_id": pid, "p": recipes[pid],
                    "bridge": bridge, "kappa": kappa, "base_losses_and_mean": base[index].tolist(),
                    "bridged_losses_and_mean": final[index].tolist()})
    return checks, values


def nested_blockers(entries):
    """Include cross-registry aliases in the same mathematical scenario."""
    groups = defaultdict(list)
    for e in entries:
        if e["state"] == "available":
            groups[canonical([e["protocol_conditions"], e["job"]["bridge"], e["job"]["kappa"]])].append(e)
    violations = []
    for values in groups.values():
        for ref, candidate in NESTED:
            for a in (e for e in values if e["job"]["mode"] == ref):
                for b in (e for e in values if e["job"]["mode"] == candidate):
                    gap = b["result"]["mean_loss"]-a["result"]["mean_loss"]
                    if gap > NESTED_ATOL:
                        violations.append({"reference_id": a["entry_id"], "candidate_id": b["entry_id"],
                            "reference_mode": ref, "candidate_mode": candidate, "loss_gap": gap,
                            "absolute_tolerance": NESTED_ATOL, "publication_blocked": True})
    return violations


def robust_analysis(entries, ev, include_zero, primary_registry="bridge"):
    groups, exclusions = {}, []
    # The primary bridge registry defines the required comparison conditions.
    # Sensitivity and dense-budget registries do not create bridge obligations.
    primary_conditions = {
        canonical(ev.conditions(e["job"], True)) for e in entries
        if e["registry"] == primary_registry and e["job"]["config"].get("q0_override") is None
        and (e["job"]["bridge"], e["job"]["kappa"]) in SCENARIOS}
    for e in entries:
        j = e["job"]
        if e["registry"] not in (primary_registry, "main", "negative"):
            exclusions.append({"entry_id": e["entry_id"], "applicability": "not_applicable",
                               "reason": "sensitivity/refinement registry; no positive-six coverage obligation"})
            continue
        if j["config"].get("q0_override") is not None or ((j["bridge"], j["kappa"]) not in SCENARIOS and not (include_zero and j["kappa"] == 0)):
            exclusions.append({"entry_id": e["entry_id"], "applicability": "not_applicable",
                               "reason": "outside default-Q0 positive-six scenario set or unrequested zero-effect stress"})
            continue
        if e["registry"] == "negative" and j["kappa"] != 0:
            exclusions.append({"entry_id": e["entry_id"], "applicability": "not_applicable",
                               "reason": "negative registry is a stress source only"})
            continue
        conditions = ev.conditions(j, True)
        key = canonical(conditions)
        if key not in primary_conditions:
            exclusions.append({"entry_id": e["entry_id"], "applicability": "not_applicable",
                               "reason": "no matching condition in the selected primary bridge registry"})
            continue
        group = groups.setdefault(key, {"conditions": conditions, "representative": j, "entries": []})
        group["entries"].append(e)
    output = {n: [] for n in ("robust_candidates", "robust_coverage", "robust_cross_evaluation", "robust_regret",
                             "ranking_checks", "ranking_values", "kappa0_stress_cross_evaluation", "kappa0_stress_regret")}
    output["robust_exclusions"] = exclusions
    for key, group in groups.items():
        gid, slots = identity(key), []
        for bridge, kappa in SCENARIOS:
            for mode in MODES:
                found = [e for e in group["entries"] if (e["job"]["bridge"], e["job"]["kappa"], e["job"]["mode"]) == (bridge, kappa, mode)]
                slots.append({"bridge": bridge, "kappa": kappa, "mode": mode,
                    "records": [{"entry_id": e["entry_id"], "state": e["state"]} for e in found],
                    "state": "available" if found and all(e["state"] == "available" for e in found) else "incomplete" if found else "not_registered"})
        pool, zero_pool = {}, {}
        for e in group["entries"]:
            if e["state"] != "available":
                continue
            selected = zero_pool if e["job"]["kappa"] == 0 else pool
            sources = [(e["result"], e["job"]["mode"], "selected_result")]
            if e.get("embedded_baseline"):
                sources.append((e["embedded_baseline"]["result"], MODES[0], "embedded_baseline"))
            for result, mode, origin in sources:
                config = {k: result[k] for k in ("N_params", "D_tokens", "Q", "p")}
                cid = identity(config)
                item = selected.setdefault(cid, {"candidate_id": cid, "result": config, "origins": [], "includes_baseline": False})
                item["origins"].append({"entry_id": e["entry_id"], "mode": mode, "origin": origin})
                item["includes_baseline"] |= mode == MODES[0]
        candidates = list(pool.values())
        output["robust_candidates"].extend({"group_id": gid, **c} for c in candidates)
        cells, summaries, common = cross_evaluate(candidates, group["representative"], SCENARIOS, ev)
        positive_entries = [e for e in group["entries"] if e["job"]["kappa"] > 0]
        violations = nested_blockers(positive_entries)
        mismatches = [c for c in duplicate_checks(positive_entries, ev) if c["state"] == "scientific_mismatch"]
        baseline = any(c["includes_baseline"] for c in common)
        complete = all(s["state"] == "available" for s in slots)
        label = "computed_finite_candidate_set" if complete and baseline and not violations and not mismatches else "provisional_incomplete_candidate_set"
        if violations or mismatches:
            label = "blocked_numerical_inconsistency"
        output["robust_coverage"].append({"group_id": gid, "conditions": group["conditions"], "state": label, "slots": slots,
            "expected_scenario_modes": 24, "available_scenario_modes": sum(s["state"] == "available" for s in slots),
            "unique_candidates": len(candidates), "common_feasible_candidates": len(common), "rejected_candidates": len(candidates)-len(common),
            "baseline_in_common_set": baseline, "nested_blockers": violations, "duplicate_mismatches": mismatches,
            "publication_blocked": label != "computed_finite_candidate_set",
            "minimax_scope": "finite found common-feasible candidate set only; not a continuous/global robust optimum"})
        output["robust_cross_evaluation"].extend({"group_id": gid, **r} for r in cells)
        output["robust_regret"].extend({"group_id": gid, "coverage_state": label, "publication_blocked": label != "computed_finite_candidate_set", **r} for r in summaries)
        checks, values = ranking(common, group["representative"], ev)
        output["ranking_checks"].extend({"group_id": gid, **r} for r in checks)
        output["ranking_values"].extend({"group_id": gid, **r} for r in values)
        if include_zero:
            combined = copy.deepcopy(pool)
            for cid, c in zero_pool.items():
                if cid in combined:
                    combined[cid]["origins"].extend(c["origins"])
                    combined[cid]["includes_baseline"] |= c["includes_baseline"]
                else:
                    combined[cid] = c
                output["robust_candidates"].append({"group_id": gid, "stress_only": True, **c})
            cells0, summaries0, _ = cross_evaluate(list(combined.values()), group["representative"], [("ratio", 0.), ("additive", 0.)], ev)
            output["kappa0_stress_cross_evaluation"].extend({"group_id": gid, "scope": "separate kappa=0 stress; excluded from positive-six minimax", **r} for r in cells0)
            output["kappa0_stress_regret"].extend({"group_id": gid, **r} for r in summaries0)
    return output


def analyze(args):
    source = Path(args.source_root).resolve()
    if source not in (ROOT, V001.resolve()):
        raise ValueError("source must be this version or the explicit V001 compatibility source")
    if source != ROOT and args.purpose != "compatibility":
        raise ValueError("V001 requires --purpose compatibility")
    if args.fixture_root and (args.purpose != "fixture" or source != ROOT):
        raise ValueError("fixtures require the current version and --purpose fixture")
    if args.purpose == "fixture" and not args.fixture_root:
        raise ValueError("fixture purpose requires --fixture-root")
    if len(set(args.registries)) != len(args.registries) or not all(re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in [*args.registries, args.robust_registry]):
        raise ValueError("registry names must be unique safe names")
    data_root = under(args.fixture_root, QA) if args.fixture_root else source / "results"
    output_root = Path(args.output_root or (QA / "runs" if args.purpose != "formal-analysis" else ROOT / "results/analysis")).resolve()
    if not any(output_root.is_relative_to(r.resolve()) for r in (ROOT / "results/analysis", QA)):
        raise ValueError("output-root outside permitted analysis directories")
    if args.purpose != "formal-analysis" and not output_root.is_relative_to(QA.resolve()):
        raise ValueError("fixture and compatibility outputs must remain in analysis QA")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
    target = output_root / f"evidence_{args.purpose}_{source.parent.name}_{'_'.join(args.registries)}" / stamp
    target.mkdir(parents=True, exist_ok=False)
    inputs = Inputs(source)
    summary = {"state": "initializing", "purpose": args.purpose, "source_version": source.parent.name,
        "source_root": str(source), "output_directory": str(target), "requested_registries": args.registries,
        "analysis_script": str(Path(__file__).resolve()), "analysis_script_sha256": digest(Path(__file__)),
        "command": sys.argv, "python": sys.executable, "python_version": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "is_review_or_approval": False,
        "float_io": "JSON authoritative; CSV %.17g / pandas round_trip",
        "scope": "conditional predictions of registered found candidates; no full optimization or global certificate",
        "publication_blocked": True, "analysis_checks_block_publication": True,
        "primary_bridge_registry": args.robust_registry,
        "admission_scope": "does not override independent M1/P1/P2 decisions"}
    try:
        ev = Evaluator(source, inputs)
        entries, issues, registries = load_entries(args.registries, data_root, bool(args.fixture_root), ev, inputs)
        inventory = [{k: v for k, v in e.items() if k in
            ("entry_id", "registry", "batch", "job_id", "job", "state", "source_file", "input_status", "error")} for e in entries]
        # Persist the denominator even if later numerical analysis encounters an error.
        table(target, "inventory", inventory)
        table(target, "issues", issues)
        configs, starts = evidence_rows(entries, ev)
        pairs, domains, modes, sensitivity = pairing(entries, ev)
        duplicates = duplicate_checks(entries, ev)
        violations = nested_blockers(entries)
        robust_applicable = args.robust_registry in args.registries
        robust = robust_analysis(entries, ev, args.include_kappa0, args.robust_registry)
        robust_scope = []
        for name in args.registries:
            role = "primary_positive_bridge" if name == args.robust_registry else "baseline_and_main_candidates" if name == "main" else "separate_zero_effect_stress" if name == "negative" else "sensitivity_or_refinement"
            applicability = "required" if name == args.robust_registry else "candidate_source" if name == "main" and robust_applicable else "stress_only" if name == "negative" and robust_applicable and args.include_kappa0 else "not_applicable"
            robust_scope.append({"registry": name, "role": role, "positive_six_applicability": applicability,
                                 "six_scenario_coverage_required": name == args.robust_registry,
                                 "own_registration_and_nested_checks_required": True})
        products = {"inventory": inventory, "issues": issues, "configurations": configs, "starts": starts,
                    "mode_pairs": pairs, "domain_pairs": domains, "four_mode_gains": modes,
                    "same_problem_numerical_differences": sensitivity, "duplicate_comparisons": duplicates,
                    "nested_publication_blockers": violations, "robust_scope": robust_scope, **robust}
        for name, rows in products.items():
            for row in rows:
                row.update(analysis_purpose=args.purpose, source_version=source.parent.name)
                if args.purpose != "formal-analysis":
                    row["publication_blocked"] = True
            table(target, name, rows)
        blocking_issues = [i for i in issues if i.get("severity") != "information"]
        complete = bool(entries) and not blocking_issues and all(e["state"] == "available" for e in entries)
        robust_complete = (bool(robust["robust_coverage"]) and all(r["state"] == "computed_finite_candidate_set" for r in robust["robust_coverage"])) if robust_applicable else None
        ranking_failures = sum(r.get("ordering_identity_holds") is False for r in robust["ranking_checks"])
        mismatches = sum(r["state"] == "scientific_mismatch" for r in duplicates)
        finished = complete and robust_complete is not False and not violations and not ranking_failures and not mismatches
        summary.update(state="completed" if finished else "incomplete", registry_inventory=registries,
            source_manifest=ev.manifest, expected_records=len(entries), record_states=dict(Counter(e["state"] for e in entries)),
            issue_count=len(issues), blocking_issue_count=len(blocking_issues), nested_violation_count=len(violations),
            ranking_failure_count=ranking_failures, duplicate_scientific_mismatch_count=mismatches,
            maximum_paired_domain_degradation=max([r.get("maximum_domain_degradation", 0.) for r in pairs], default=None),
            positive_six_scenario_groups=len(robust["robust_coverage"]), positive_six_coverage_complete=robust_complete,
            positive_six_applicability="required" if robust_applicable else "not_applicable",
            registered_results_complete=complete, analysis_checks_block_publication=not finished,
            publication_blocked=not finished or args.purpose != "formal-analysis", table_rows={k: len(v) for k, v in products.items()},
            analytical_ranking_argument={
                "ratio": "At fixed N,D,Q,Q0, a=(F7(Q)/F7(Q0))**kappa>0 is independent of p: Delta L3=a*Delta Lmix.",
                "additive": "At fixed N,D,Q,Q0, b=kappa*(F7(Q)-F7(Q0)) is independent of p: Delta L3=Delta Lmix.",
                "scope": "Each domain and equal-weight mean preserve recipe order on common feasible recipes.",
                "coupling": "Joint nonadditivity can reflect budget sharing and scale reoptimization, not identified direct Q-by-p synergy.",
                "numeric_probes": "Baseline and largest-Q found candidate N/D/Q per group; all common candidate recipes plus A4 reference."})
        if ev.base.source_manifest() != ev.manifest:
            summary["manifest_changed_at_finish"] = True
    except Exception as exc:
        summary.update(state="error", error_type=type(exc).__name__, error=str(exc), publication_blocked=True,
                       analysis_checks_block_publication=True)
    summary["input_sha256"] = inputs.hashes
    summary["changed_inputs_at_finish"] = inputs.changed()
    summary["analysis_script_unchanged"] = digest(Path(__file__)) == summary["analysis_script_sha256"]
    if summary["changed_inputs_at_finish"] or not summary["analysis_script_unchanged"] or summary.get("manifest_changed_at_finish"):
        summary.update(state="invalidated_input_change", publication_blocked=True, analysis_checks_block_publication=True)
    summary["output_sha256"] = {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file()}
    write_json(target / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("state", "purpose", "source_version", "output_directory")}, ensure_ascii=False))
    return 0 if summary["state"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--registries", nargs="+", default=["main", "bridge"])
    parser.add_argument("--robust-registry", default="bridge",
                        help="primary bridge registry defining positive-six conditions; if unselected, robustness is N/A")
    parser.add_argument("--purpose", choices=("formal-analysis", "compatibility", "fixture"), default="formal-analysis")
    parser.add_argument("--fixture-root")
    parser.add_argument("--output-root")
    parser.add_argument("--include-kappa0", action="store_true")
    return analyze(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
