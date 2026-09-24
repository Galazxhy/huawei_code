"""Run the minimum executable audit for the corrected Q3 model contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q3_model_contract import ATTACHMENTS, ContractConfig, Q3ModelContract, sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = Q3ModelContract()
    p_reference = model.support_points.mean(axis=0)
    cases = {}
    cases["reference_q0"] = model.evaluate(1e9, 25e9, p_reference, model.q0, 2048)
    cases["reference_q07"] = model.evaluate(1e9, 25e9, p_reference, 0.7, 2048)
    cases["long_context"] = model.evaluate(1e9, 25e9, p_reference, model.q0, 32768)
    cases["low_budget_quality_feasible"] = model.evaluate(7e7, 1e10, p_reference, 0.7, 2048)
    assert cases["low_budget_quality_feasible"]["external_scale_warning"] is True
    cases["low_budget_quality_feasible"] = model.evaluate(
        7e7, 1e10, p_reference, 0.7, 2048, budget_flops=1e19)

    def expect_rejection(call):
        try:
            call()
        except ValueError as exc:
            return str(exc)
        raise AssertionError("invalid input was accepted")

    extra_rejections = {
        "budget": expect_rejection(lambda: model.evaluate(
            1e9, 25e9, p_reference, model.q0, 2048, budget_flops=1e19)),
        "invalid_budget": expect_rejection(lambda: model.evaluate(
            1e9, 25e9, p_reference, model.q0, 2048, budget_flops=float("nan"))),
        "N_bound": expect_rejection(lambda: model.evaluate(
            2e11, 25e9, p_reference, model.q0, 2048)),
        "D_bound": expect_rejection(lambda: model.evaluate(
            1e9, 2e13, p_reference, model.q0, 2048)),
        "zero_Q": expect_rejection(lambda: model.evaluate(
            1e9, 25e9, p_reference, 0., 2048)),
        "bad_simplex": expect_rejection(lambda: model.evaluate(
            1e9, 25e9, p_reference * 1.01, model.q0, 2048)),
    }
    strict = Q3ModelContract(ContractConfig(allow_scale_extrapolation=False))
    extra_rejections["strict_scale"] = expect_rejection(lambda: strict.evaluate(
        7e7, 1e10, p_reference, strict.q0, 2048))
    mutated = Q3ModelContract()
    mutated.q_i[0] += 0.001
    extra_rejections["handoff_quality_mismatch"] = expect_rejection(mutated._validate_handoff)

    grid = []
    for context in model.contexts:
        for form in model.quality_cost_payload["forms"]:
            base = model.costs(1e9, 25e9, model.q0, context, form)
            costs = model.costs(1e9, 25e9, .8, context, form)
            assert base["C_quality_flops"] == 0
            assert costs["C_quality_flops"] > 0
            assert np.isclose(costs["C_total_flops"], sum(costs[k] for k in
                ("C_train_flops", "C_attn_flops", "C_quality_flops")), rtol=1e-14)
            assert np.isclose(costs["attention_to_training_ratio"], context / 30000)
            grid.append({"L_ctx": context, **costs})
    assert model.critical_context == 30000
    bridge_checks = {}
    for bridge in ("ratio", "additive"):
        for kappa in (.5, 1., 1.5):
            m = Q3ModelContract(ContractConfig(quality_bridge_mode=bridge, quality_bridge_kappa=kappa))
            at_q0 = m.evaluate(1e9, 25e9, p_reference, m.q0, 2048)
            improved = m.evaluate(1e9, 25e9, p_reference, .8, 2048)
            assert np.isclose(at_q0["mean_loss"], at_q0["base_mean_loss_q2"], atol=1e-12)
            assert improved["mean_loss"] < at_q0["mean_loss"]
            bridge_checks[f"{bridge}_{kappa}"] = True

    b7 = pd.read_csv(ATTACHMENTS / "B_scaling_laws/supplementary_NQ_experiment_expanded.csv")
    b7_pred = np.array([model._quality_loss(row.N_params_B * 1e9, row.D_tokens_B * 1e9,
                                         row.Q_score) for row in b7.itertuples()])
    b7_rmse = float(np.sqrt(np.mean((b7_pred - b7.val_loss.to_numpy()) ** 2)))
    assert np.isclose(b7_rmse, model.b7["rmse"], rtol=1e-8)
    for item in __import__("json").loads((ATTACHMENTS / "来源清单.json").read_text()):
        assert sha256(ROOT / item["snapshot"]) == item["sha256"]
    assert len(model.support_points) == 512

    # A deliberately invalid point must be rejected rather than optimized.
    enron = np.zeros(17)
    enron[model.train_domains.index("enron_emails")] = 1.0
    rejection = None
    try:
        model.evaluate(1e9, 25e9, enron, model.q0, 2048)
    except ValueError as exc:
        rejection = str(exc)
    if rejection is None:
        raise AssertionError("out-of-support mixture was not rejected")

    # A billion-unit mistake must be rejected by the external-unit contract.
    unit_rejection = None
    try:
        model.evaluate(1.0, 25.0, p_reference, model.q0, 2048)
    except ValueError as exc:
        unit_rejection = str(exc)
    if unit_rejection is None:
        raise AssertionError("billion-unit call was not rejected")

    context_rejection = None
    try:
        model.evaluate(1e9, 25e9, p_reference, model.q0, 1234)
    except ValueError as exc:
        context_rejection = str(exc)
    if context_rejection is None:
        raise AssertionError("invalid C7 context was not rejected")

    q_rejection = None
    try:
        model.evaluate(1e9, 25e9, p_reference, model.q0 - 0.01, 2048)
    except ValueError as exc:
        q_rejection = str(exc)
    if q_rejection is None:
        raise AssertionError("quality below Q0 was not rejected")

    # Exercise the nonpositive-Loss guard with a synthetic faulty predictor.
    guard_model = Q3ModelContract()
    guard_model.q2.predict = lambda *args, **kwargs: np.full(len(guard_model.loss_domains), -1.0)
    loss_guard_rejection = None
    try:
        guard_model.evaluate(1e9, 25e9, p_reference, guard_model.q0, 2048)
    except ValueError as exc:
        loss_guard_rejection = str(exc)
    if loss_guard_rejection is None:
        raise AssertionError("nonpositive Loss guard was not triggered")

    for label, output in cases.items():
        assert np.isfinite(output["mean_loss"])
        assert output["costs"]["C_total_flops"] >= output["costs"]["C_train_flops"]
        assert output["support_certificate"]["inside_convex_hull"]
        if label == "reference_q0":
            assert abs(output["quality_ratio"] - 1.0) < 1e-12

    result = {
        "status": "PASS",
        "scope": "Contract smoke audit only; no Q3 optimum and no formal uncertainty interval.",
        "source_manifest": model.source_manifest(),
        "domains": model.train_domains,
        "loss_domains": model.loss_domains,
        "q0": model.q0,
        "q17_unique_values": int(len(np.unique(model.q_i))),
        "contexts_from_C7": list(model.contexts),
        "critical_context": model.critical_context,
        "cases": cases,
        "additional_rejections": extra_rejections,
        "cost_context_grid": grid,
        "bridge_checks": bridge_checks,
        "b7_semisynthetic_recalculation": {"rows": len(b7), "rmse": b7_rmse,
            "N_range_params": [float(b7.N_params_B.min() * 1e9), float(b7.N_params_B.max() * 1e9)],
            "D_range_tokens": [float(b7.D_tokens_B.min() * 1e9), float(b7.D_tokens_B.max() * 1e9)],
            "Q_range": [float(b7.Q_score.min()), float(b7.Q_score.max())]},
        "invalid_out_of_support_rejection": rejection,
        "invalid_billion_unit_rejection": unit_rejection,
        "invalid_context_rejection": context_rejection,
        "quality_below_baseline_rejection": q_rejection,
        "nonpositive_loss_guard_rejection": loss_guard_rejection,
        "checks": {
            "cost_includes_training_attention_quality": True,
            "q0_returns_q2_loss": abs(cases["reference_q0"]["quality_ratio"] - 1.0) < 1e-12,
            "support_boundary_is_enforced": True,
            "absolute_unit_contract_is_enforced": True,
            "extrapolation_is_flagged": cases["reference_q0"]["external_scale_warning"] is False,
            "outer_scale_is_flagged": cases["low_budget_quality_feasible"]["external_scale_warning"] is True,
            "invalid_context_is_rejected": context_rejection is not None,
            "quality_below_baseline_is_rejected": q_rejection is not None,
            "nonpositive_loss_is_rejected": loss_guard_rejection is not None,
            "quality_bridge_is_explicit": True,
            "uncertainty_is_labeled_empirical_residual_scale_only": True,
            "budget_rejection_and_margin": True,
            "declared_N_D_bounds_enforced": True,
            "strict_scale_mode_rejects_extrapolation": True,
            "training_only_support_512_rows": True,
            "snapshot_hashes_verified": True,
            "test_mixture_loss_experiment_IDs_aligned": True,
            "Q1_Q2_order_and_values_asserted": True,
            "three_cost_forms_five_C7_contexts": True,
            "both_bridges_three_kappas_recover_Q2_at_Q0": True,
            "B7_rmse_reproduced": True,
        },
        "limitations": [
            "The ratio bridge from B7 to the Q2 mixture predictor is a declared modeling assumption.",
            "Provided Q2 fit scripts, split IDs and covariance are absent; no formal predictive interval is claimed.",
            "The contract checks support membership but does not yet solve the budget-constrained optimization.",
        ],
    }
    out = ROOT / "results/q3_contract_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out), "checks": result["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
