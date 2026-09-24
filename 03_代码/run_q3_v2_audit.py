"""Meaningful regression checks for V002 controls and numerical diagnostics."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json

import numpy as np

from q3_model_contract import ROOT, ContractConfig, Q3ModelContract
from q3_optimizer import JointOptimizer, OptimizerConfig
from q3_structure import candidates
from run_q3_experiments import run_job, write_json


def main():
    checks = {}
    model = Q3ModelContract()
    reference = model.support_points.mean(axis=0)
    for bridge in ("ratio", "additive"):
        null = Q3ModelContract(ContractConfig(quality_bridge_mode=bridge, quality_bridge_kappa=0))
        values = [null.predict_losses(1e9, 25e9, reference, q)[0] for q in (null.q0, .8, 1.)]
        assert all(np.array_equal(values[0], value) for value in values[1:])
        assert null.costs(1e9, 25e9, 1., 4096, "exponential")["C_quality_flops"] > 0
    checks["null_quality_effect_keeps_cost_but_removes_loss_benefit"] = True
    try:
        Q3ModelContract(ContractConfig(quality_bridge_kappa=-1))
        raise AssertionError("negative transfer accepted")
    except ValueError:
        checks["negative_transfer_rejected"] = True

    points = model.support_points
    weighted = points*model.q_i
    nonconstant = points.std(axis=0) > 0
    assert np.allclose(((weighted-weighted.mean(axis=0))/weighted.std(axis=0))[:, nonconstant],
                       ((points-points.mean(axis=0))/points.std(axis=0))[:, nonconstant], atol=1e-12)
    checks["fixed_positive_domain_quality_does_not_identify_independent_quality"] = True
    base = np.array([model.q2.predict(1e9, 25e9, p).mean() for p in points])
    difference = base[:, None]-base[None, :]
    for bridge in ("ratio", "additive"):
        other = Q3ModelContract(ContractConfig(quality_bridge_mode=bridge))
        values = np.array([other.predict_losses(1e9, 25e9, p, .8)[0].mean() for p in points])
        changed = values[:, None]-values[None, :]
        assert np.all(np.sign(difference[np.abs(difference)>1e-10]) == np.sign(changed[np.abs(difference)>1e-10]))
    checks["both_bridges_preserve_fixed_scale_mixture_ranking"] = True
    point = model.evaluate(1e9, 25e9, reference, model.q0, 4096)
    assert point["nearest_scale_log10_distance"] == 0
    assert point["evidence_levels"]["scale_response"] == "observed scale"
    checks["observed_anchor_has_zero_scale_distance"] = True

    result = JointOptimizer(bridge_mode="ratio", kappa=0).solve_mode(1e22, 4096, "exponential")
    no_governance = JointOptimizer(bridge_mode="ratio", kappa=0).solve_mode(1e22, 4096, "exponential", "fixed_q")
    assert result["Q"]-model.q0 < 1e-4
    assert abs(result["mean_loss"]-no_governance["mean_loss"]) < 1e-6
    assert all("p" in a and "model_evaluations" in a for a in result["attempts"])
    assert result["feasible_perturbation_pass"]
    checks["zero_effect_solver_does_not_purchase_quality"] = True
    checks["actual_solver_has_final_candidates_and_feasible_probes"] = True

    rows = [{"budget_flops": 10.**(19+i), "N_params": 1e7*2**i, "D_tokens": 1e10*2**i,
             "Q": .6, "Q0": .6, "p": [1/17]*17, "active_constraints": ["budget"],
             "cost_shares": {"quality": .18*i, "train": 1-.18*i}} for i in range(5)]
    events = candidates(rows)["events"]
    assert events and all(e["channel"] == "cost_share" and not e["mechanism_verified"] for e in events)
    checks["smooth_share_change_is_not_mechanism_evidence"] = True

    job = {"bridge": "ratio", "kappa": 1., "config": asdict(OptimizerConfig()),
           "scenario": "parallel_execution_audit", "budget": 1e19, "context": 2048,
           "form": "exponential", "mode": "fixed_p_q_baseline"}
    manifest = model.source_manifest()
    serial = run_job(job, manifest)
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = [f.result() for f in [pool.submit(run_job, job, manifest) for _ in range(2)]]
    for record in parallel:
        assert record["status"] == serial["status"] == "PASS"
        assert all(np.allclose(record["result"][key], serial["result"][key], rtol=1e-12, atol=1e-12)
                   for key in ("N_params", "D_tokens", "Q", "p", "mean_loss"))
    checks["independent_workers_reproduce_serial_numerical_configuration"] = True
    write_json(ROOT / "results/q3_v2_audit.json", {"status": "PASS", "checks": checks,
               "source_manifest": manifest, "scope": "targeted controls, not full formal matrix"})
    print(json.dumps({"status": "PASS", "checks": checks}))


if __name__ == "__main__":
    main()
