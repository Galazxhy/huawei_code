"""Mechanism-specific transition candidates and scenario stability, not CIs."""
import argparse
import itertools
import json
import hashlib
from collections import defaultdict

import numpy as np

from q3_model_contract import ROOT, Q3ModelContract
from run_q3_experiments import job_key, load_registry, write_json


def quality_state(result, tolerance):
    if result["Q"] <= result["Q0"] + tolerance:
        return "lower"
    if result["Q"] >= 1 - tolerance:
        return "upper"
    return "interior"


def quality_direction_check(result, state, tolerance=1e-4):
    direction = result.get("quality_budget_directional_derivative")
    applicable = result.get("quality_direction_valid_for_fixed_N_p", False)
    if not applicable or direction is None or not np.isfinite(direction):
        return False
    return bool(direction >= -tolerance if state == "lower" else
                direction <= tolerance if state == "upper" else abs(direction) <= tolerance)


def near_equal_support_ambiguity(result, tau, relative_gap=1e-4):
    selected = set(np.flatnonzero(np.asarray(result["p"]) > tau))
    objective = result.get("mean_loss")
    if objective is None:
        return None
    for attempt in result.get("attempts", []):
        if attempt.get("accepted") and attempt.get("objective", np.inf) <= objective + relative_gap*max(abs(objective), 1.):
            if set(np.flatnonzero(np.asarray(attempt["p"]) > tau)) != selected:
                return True
    return False


def candidates(results, tau=.01, delta_p=.2, delta_c=.15, q_tol=.001,
               direction_tol=1e-4, near_equal_gap=1e-4):
    results = sorted(results, key=lambda r: r["budget_flops"])
    events, elasticities = [], []
    for i, (before, after) in enumerate(zip(results, results[1:])):
        left, right = before["budget_flops"], after["budget_flops"]
        if left == right:
            raise ValueError("duplicate budget in trajectory")
        elasticities.append({"interval": [left, right], **{
            key: float(np.log(after[key] / before[key]) / np.log(right / left))
            for key in ("N_params", "D_tokens")}})
        s0, s1 = set(np.flatnonzero(np.array(before["p"]) > tau)), set(np.flatnonzero(np.array(after["p"]) > tau))
        jac = 1 - len(s0 & s1) / len(s0 | s1) if s0 | s1 else 0.
        mixture_l1 = float(np.abs(np.asarray(before["p"])-np.asarray(after["p"])).sum())
        costs = set(before["cost_shares"])
        cost_delta = max(abs(after["cost_shares"][k] - before["cost_shares"][k]) for k in costs)
        changed_costs = [k for k in costs if abs(after["cost_shares"][k] - before["cost_shares"][k]) >= delta_c]
        state0, state1 = quality_state(before, q_tol), quality_state(after, q_tol)
        boundary0 = {x for x in before["active_constraints"] if x.startswith(("N_", "D_"))}
        boundary1 = {x for x in after["active_constraints"] if x.startswith(("N_", "D_"))}
        triggered = []
        if state0 != state1:
            triggered.append("quality_regime")
        if jac >= delta_p:
            triggered.append("mixture_support")
        if cost_delta >= delta_c:
            triggered.append("cost_share")
        if boundary0 != boundary1:
            triggered.append("scenario_boundary")
        future = results[i + 2:i + 4]
        for channel in triggered:
            verified = False
            if channel == "quality_regime":
                mechanism = {"from": state0, "to": state1}
                verified = quality_direction_check(before, state0, direction_tol) and quality_direction_check(after, state1, direction_tol)
            elif channel == "mixture_support":
                mechanism = {"entered": [int(x) for x in sorted(s1 - s0)],
                             "exited": [int(x) for x in sorted(s0 - s1)]}
                verified = (mixture_l1 >= .02 and near_equal_support_ambiguity(before, tau, near_equal_gap) is False
                            and near_equal_support_ambiguity(after, tau, near_equal_gap) is False
                            and before.get("feasible_perturbation_pass", False)
                            and after.get("feasible_perturbation_pass", False))
            elif channel == "scenario_boundary":
                mechanism = {"entered": sorted(boundary1 - boundary0), "exited": sorted(boundary0 - boundary1)}
                verified = True
            else:
                mechanism = {k: int(np.sign(after["cost_shares"][k] - before["cost_shares"][k]))
                             for k in sorted(changed_costs)}
            numerical_ok = all(r.get("feasible_perturbation_pass", False) for r in (before, after))
            verified = verified and numerical_ok
            persisted = None
            if len(future) == 2:
                if channel == "quality_regime":
                    persisted = all(quality_state(r, q_tol) == state1 for r in future)
                    verified = verified and all(quality_direction_check(r, state1, direction_tol)
                                                 and r.get("feasible_perturbation_pass", False) for r in future)
                elif channel == "mixture_support":
                    persisted = all(set(np.flatnonzero(np.array(r["p"]) > tau)) == s1 for r in future)
                elif channel == "scenario_boundary":
                    persisted = all({x for x in r["active_constraints"] if x.startswith(("N_", "D_"))} == boundary1 for r in future)
                else:
                    persisted = all(all(
                        abs(r["cost_shares"][k] - before["cost_shares"][k]) >= delta_c and
                        np.sign(r["cost_shares"][k] - before["cost_shares"][k]) ==
                        np.sign(after["cost_shares"][k] - before["cost_shares"][k])
                        for r in future) for k in changed_costs)
            events.append({"interval": [left, right], "channel": channel,
                           "mechanism": mechanism,
                           "mechanism_verified": bool(verified),
                           "numerical_qualification": numerical_ok,
                           "effective_support_threshold_not_true_zero_support": channel == "mixture_support",
                           "cost_share_change_alone_is_descriptive": channel == "cost_share",
                           "mixture_l1_change": mixture_l1,
                           "near_equal_support_ambiguity_before": near_equal_support_ambiguity(before, tau, near_equal_gap),
                           "near_equal_support_ambiguity_after": near_equal_support_ambiguity(after, tau, near_equal_gap),
                           "persistent_two_future_points": persisted, "jaccard": jac,
                           "max_cost_share_change": cost_delta,
                           "quality_direction_after": after.get("quality_budget_directional_derivative"),
                           "direction_applicable": after.get("quality_direction_valid_for_fixed_N_p", False)})
    return {"events": events, "resource_elasticities": elasticities,
            "thresholds": {"tau": tau, "delta_p": delta_p, "delta_c": delta_c, "q_tol": q_tol,
                           "direction_tolerance": direction_tol, "near_equal_relative_gap": near_equal_gap}}


def trajectory_id(job):
    return json.dumps({k: v for k, v in job.items() if k not in ("scenario", "budget")}, sort_keys=True)


def same_mechanism(event, channel, mechanism):
    return event["channel"] == channel and event["mechanism"] == mechanism


def accepted_candidate_signature(result):
    """Retain the evidence used by every support/gap threshold, without timing."""
    candidates_seen = set()
    for attempt in result.get("attempts", []):
        if not attempt.get("accepted"):
            continue
        objective = float(attempt["objective"])
        mixture = np.asarray(attempt["p"], dtype=float)
        if mixture.shape != (17,) or not np.isfinite(mixture).all() or not np.isfinite(objective):
            raise ValueError("invalid accepted candidate evidence")
        candidates_seen.add((objective, tuple(mixture.tolist())))
    return tuple(sorted(candidates_seen))


def scientific_representative_key(result):
    """Choose a representative from scientific values, never path or timing."""
    keys = ("N_params", "D_tokens", "Q", "Q0", "p", "mean_loss", "loss_by_domain", "costs", "cost_shares",
            "quality_budget_directional_derivative", "quality_direction_valid_for_fixed_N_p",
            "feasible_perturbation_pass", "active_constraints", "boundary_sensitive",
            "external_scale_warning", "optimization_mode", "solver_success")
    return json.dumps({key: result[key] for key in keys if key in result}, sort_keys=True, allow_nan=False)


def scientific_equivalence(left, right):
    """Compare selected science and the complete accepted qualification evidence."""
    keys = ("N_params", "D_tokens", "Q", "Q0", "p", "mean_loss", "loss_by_domain", "costs", "cost_shares",
            "quality_budget_directional_derivative", "quality_direction_valid_for_fixed_N_p",
            "feasible_perturbation_pass", "active_constraints", "boundary_sensitive",
            "external_scale_warning", "optimization_mode", "solver_success")

    def equal(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return bool(np.isclose(a, b, rtol=1e-9, atol=1e-10))
        return a == b
    if not all((k in left) == (k in right) and equal(left.get(k), right.get(k)) for k in keys):
        return False
    # These values enter discontinuous thresholds, sometimes with a neighbouring
    # budget. Numeric closeness alone cannot certify equivalent mechanism votes.
    threshold_keys = ("Q", "Q0", "p", "mean_loss", "cost_shares",
                      "quality_budget_directional_derivative")
    if any(left.get(k) != right.get(k) for k in threshold_keys):
        return False
    # Threshold decisions are discontinuous: do not round distinct candidates away.
    try:
        return accepted_candidate_signature(left) == accepted_candidate_signature(right)
    except (KeyError, TypeError, ValueError):
        return False


def refinement_evidence(low, high, channel, mechanism, group, registries, observed):
    """Require a narrower persistent event inside an original main event interval."""
    main = next((r for r in registries if r["batch"] == "main" and
                 r["jobs"] and all(j.get("scenario") == "main" for j in r["jobs"]) and
                 len({j["budget"] for j in r["jobs"]}) == 11), None)
    if main is None:
        return {"passed": False, "reason": "no designated eleven-budget main registry"}
    jobs = [j for j in main["jobs"] if (j["context"], j["form"], j["mode"]) == group]
    if not jobs:
        return {"passed": False, "reason": "main trajectory missing"}
    tid = trajectory_id(jobs[0])
    original_keys = {job_key(j) for j in jobs}
    extra = {job_key(j): j for r in registries for j in r["jobs"]
             if trajectory_id(j) == tid and job_key(j) not in original_keys}
    required = original_keys | set(extra)
    if not all(k in observed and observed[k]["status"] == "PASS" for k in required):
        return {"passed": False, "reason": "main or refined records incomplete/invalid"}
    coarse = candidates([observed[k]["result"] for k in original_keys])["events"]
    dense = candidates([observed[k]["result"] for k in required])["events"]
    for prior in coarse:
        pl, ph = prior["interval"]
        added = sorted(j["budget"] for j in extra.values() if pl < j["budget"] < ph)
        if not added or not same_mechanism(prior, channel, mechanism):
            continue
        if max(low, pl) >= min(high, ph):
            continue
        for refined in dense:
            rl, rh = refined["interval"]
            if (same_mechanism(refined, channel, mechanism) and pl <= rl < rh <= ph
                    and np.log(rh/rl) < np.log(ph/pl) - 1e-12
                    and max(low, rl) < min(high, rh)
                    and refined["persistent_two_future_points"] is True and refined["mechanism_verified"]):
                return {"passed": True, "main_trajectory": tid, "coarse_interval": [pl, ph],
                        "refined_interval": [rl, rh], "added_interior_budgets": added,
                        "log_width_ratio": float(np.log(rh/rl)/np.log(ph/pl)),
                        "coarse_persistence": prior["persistent_two_future_points"], "refined_persistence": True}
    return {"passed": False, "reason": "no matching within-interval narrowed persistent main event"}


def analyze():
    manifest = Q3ModelContract().source_manifest()
    registry_paths = sorted((ROOT / "results").glob("registered_*.json"))
    registries = [load_registry(path, manifest) for path in registry_paths]
    expected = {job_key(job): job for registry in registries for job in registry["jobs"]}
    if not expected:
        raise ValueError("no frozen experiment registries found")
    observed, provenance, duplicate_results = {}, defaultdict(list), defaultdict(list)
    for file in sorted((ROOT / "results/formal").glob("*/*.json")):
        if file.name == "batch_status.json":
            continue
        record = json.loads(file.read_text(encoding="utf-8"))
        if record["source_manifest"] != manifest:
            raise ValueError(f"stale evidence: {file}")
        key = job_key(record["job"])
        incoming_result = record.get("result")
        provenance[key].append({"path": str(file.relative_to(ROOT)),
                                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        if key in observed:
            if observed[key]["status"] != "PASS" or record["status"] != "PASS":
                record = {**record, "status": "FAIL", "error": "conflicting duplicate including failed result"}
            elif any(not scientific_equivalence(prior, record["result"])
                     for prior in duplicate_results[key]):
                record = {**record, "status": "FAIL", "error": "scientifically conflicting duplicate results"}
            else:
                record = min((observed[key], record), key=lambda r: scientific_representative_key(r["result"]))
        if record["status"] == "PASS":
            duplicate_results[key].append(incoming_result)
        observed[key] = record
    nested_failures = []
    for key, job in expected.items():
        record = observed.get(key)
        if job["mode"] != "joint_reduced_hull" or not record or record["status"] != "PASS":
            continue
        for mode in ("fixed_p", "fixed_q", "fixed_p_q_baseline"):
            other_key = job_key({**job, "mode": mode})
            other = observed.get(other_key)
            reason = None
            if other_key not in expected or not other or other["status"] != "PASS":
                reason = "required nested comparator missing or invalid"
            elif record["result"]["mean_loss"] > other["result"]["mean_loss"] + 1e-7:
                reason = "joint solver candidate dominated by nested comparator"
            if reason:
                nested_failures.append({"job": key, "mode": mode, "reason": reason})
        if any(f["job"] == key for f in nested_failures):
            observed[key] = {**record, "status": "FAIL", "error": "nested comparison acceptance failed"}
    missing = sorted(set(expected) - set(observed))
    failed = [k for k in expected if k in observed and observed[k]["status"] != "PASS"]
    trajectories, expected_by_group = defaultdict(list), defaultdict(set)
    expected_by_trajectory = defaultdict(set)
    for key, job in expected.items():
        group = (job["context"], job["form"], job["mode"])
        expected_by_group[group].add(trajectory_id(job))
        expected_by_trajectory[trajectory_id(job)].add(key)
        if key in observed and observed[key]["status"] == "PASS":
            trajectories[trajectory_id(job)].append(observed[key]["result"])
    profiles, report = {}, []
    for tid, rows in trajectories.items():
        profiles[tid] = candidates(rows)
    for group, tids in expected_by_group.items():
        settings = [json.loads(tid) for tid in tids]
        prerequisites = {
            "multiple_bridges": len({s["bridge"] for s in settings}) >= 2,
            "multiple_kappas": len({s["kappa"] for s in settings}) >= 2,
            "multiple_seeds": len({s["config"]["seed"] for s in settings}) >= 2,
            "multiple_bounds": len({json.dumps([s["config"]["n_bounds"], s["config"]["d_bounds"]]) for s in settings}) >= 2,
            "multiple_anchor_counts": len({s["config"]["anchor_count"] for s in settings}) >= 2,
        }
        robustness_ready = all(prerequisites.values())
        complete = all(all(k in observed and observed[k]["status"] == "PASS"
                           for k in expected_by_trajectory[tid]) for tid in tids)
        intervals = set(tuple(e["interval"]) + (e["channel"], json.dumps(e["mechanism"], sort_keys=True))
                        for tid in tids for e in profiles.get(tid, {}).get("events", []))
        for low, high, channel, mechanism_key in sorted(intervals):
            mechanism = json.loads(mechanism_key)
            refinement = refinement_evidence(low, high, channel, mechanism, group, registries, observed)
            votes = sum(any(same_mechanism(e, channel, mechanism) and e["persistent_two_future_points"] is True
                            and e["mechanism_verified"]
                            and max(low, e["interval"][0]) < min(high, e["interval"][1])
                            for e in profiles.get(tid, {}).get("events", [])) for tid in tids)
            total = len(tids)
            report.append({"context": group[0], "cost_form": group[1], "mode": group[2],
                           "interval": [low, high], "channel": channel,
                           "mechanism": mechanism,
                           "scenario_votes": votes, "registered_denominator": total,
                           "fraction": votes / total, "complete_group": complete,
                           "robustness_prerequisites": prerequisites,
                           "interval_refinement": refinement,
                           "stable_at_threshold": {str(t): complete and robustness_ready and refinement["passed"] and votes / total >= t for t in (.6, .7, .8)}})
    threshold_counts = []
    for tau, dp, dc, qt in itertools.product((.005, .01, .02), (.1, .2, .3), (.1, .15, .2), (.0001, .001, .01)):
        events = {tid: candidates(rows, tau, dp, dc, qt)["events"] for tid, rows in trajectories.items()}
        threshold_counts.append({"tau": tau, "delta_p": dp, "delta_c": dc, "q_tolerance": qt,
                                 "events_by_trajectory": events})
    numerical_thresholds = []
    for direction_tol, objective_gap in itertools.product((1e-5,1e-4,1e-3), repeat=2):
        numerical_thresholds.append({"direction_tolerance":direction_tol,"near_equal_relative_gap":objective_gap,
            "events_by_trajectory":{tid:candidates(rows,direction_tol=direction_tol,
                                    near_equal_gap=objective_gap)["events"] for tid,rows in trajectories.items()}})
    output = {"status": "INCOMPLETE" if missing or failed else "PASS",
              "scope": "conditional scenario stability; no statistical confidence interpretation",
              "expected_count": len(expected), "observed_count": len(observed),
              "registry_hashes": {str(p.relative_to(ROOT)): r["registry_sha256"] for p, r in zip(registry_paths, registries)},
              "unregistered_observed_count": len(set(observed) - set(expected)),
              "missing_count": len(missing), "failed_count": len(failed),
              "failed_jobs": failed, "trajectories": profiles, "stability": report,
              "nested_acceptance_failures": nested_failures,
              "provenance_by_configuration": dict(provenance),
              "duplicate_scientific_tolerances": {"rtol":1e-9,"atol":1e-10,"runtime_metadata_excluded":True,
                  "accepted_candidate_objective_and_p": "exact set equality; ordering and repeated identical points excluded",
                  "threshold_fields": "exact Q/Q0/p/mean_loss/cost_shares/quality_direction",
                  "representative_rule": "lexicographically least canonical JSON of selected scientific fields",
                  "group_rule": "all pairs must agree; a failed source invalidates the entire configuration"},
              "numerical_threshold_sensitivity": numerical_thresholds,
              "threshold_sensitivity": threshold_counts, "source_manifest": manifest}
    write_json(ROOT / "results/q3_structure_shift.json", output)
    print(json.dumps({k: output[k] for k in ("status", "expected_count", "observed_count", "missing_count", "failed_count")}))


def audit():
    # Fixed p with quality activation must be detectable; smooth constant states
    # must not invent a transition. End-of-grid events lack persistence evidence.
    rows = []
    for i in range(5):
        rows.append({"budget_flops": 10. ** (19+i), "N_params": 1e7*2**i, "D_tokens": 1e10*2**i,
                     "Q": .6 if i == 0 else .8, "Q0": .6, "p": [1/17]*17,
                     "cost_shares": {"train": .8, "attention": .2, "quality": 0},
                     "active_constraints": ["budget"]})
    out = candidates(rows)
    assert any(e["channel"] == "quality_regime" and e["persistent_two_future_points"] for e in out["events"])
    assert not any(e["channel"] == "mixture_support" for e in out["events"])
    constant = [{**r, "Q": .6} for r in rows]
    assert candidates(constant)["events"] == []
    last = [{**r, "Q": .6 if i < 4 else .8} for i, r in enumerate(rows)]
    assert candidates(last)["events"][0]["persistent_two_future_points"] is None
    print("PASS: fixed-p quality activation, negative control, incomplete persistence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "analyze"))
    args = parser.parse_args()
    audit() if args.action == "audit" else analyze()
