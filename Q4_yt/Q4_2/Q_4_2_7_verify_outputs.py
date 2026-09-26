"""Fail-fast verification for Q4 result contracts and reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q4_utils import ANALYSIS_END, OUT, Q3_ANCHORS


def main() -> None:
    sample_summary = json.loads((OUT / "sample_frontier_summary.json").read_text(encoding="utf-8"))
    compute_summary = json.loads((OUT / "compute_frontier_summary.json").read_text(encoding="utf-8"))
    decomposition_summary = json.loads((OUT / "decomposition_summary.json").read_text(encoding="utf-8"))
    forecast_summary = json.loads((OUT / "forecast_summary.json").read_text(encoding="utf-8"))
    contracts = pd.DataFrame(
        [
            ["C1+C2", "capability frontier", "open weights; six complete tasks; chat/finetuned", "Submission Date", sample_summary["main_sample_n"], "main"],
            ["C1+C2", "type sensitivity", "open weights; pretrained", "Submission Date", sample_summary["pretrained_sample_n"], "sensitivity"],
            ["C3", "long-run magnitude check", "historical papers/reports only", "Year", forecast_summary["c3_long_run_check"]["records"], "diagnostic"],
            ["C4", "compute frontier", "open-weight Language LM/QA; reported compute; Confident/Likely", "Publication date", compute_summary["reported_trusted_main"], "main"],
            ["C4", "compute sensitivity", "reported plus 6ND-estimated", "Publication date", compute_summary["reported_compute"] + compute_summary["estimated_only"], "sensitivity"],
            ["C6", "Loss-Benchmark bridge", "High and Medium retained with separate level/variance", "not used", decomposition_summary["bridge_n_high"] + decomposition_summary["bridge_n_medium"], "main"],
            ["C8", "per-task breadth", "latest parseable snapshot; six task metrics", "evaluation timestamp", sample_summary["c8_complete_models"], "required diagnostic"],
            ["Q3", "compute-to-optimal-Loss mechanism", "main; context 4096; three cost forms", "budget", 3, "mechanism anchors"],
        ],
        columns=["source", "role", "filters", "time_axis", "n", "evidence_role"],
    )
    contracts.to_csv(OUT / "data_contract.csv", index=False, encoding="utf-8-sig")
    required = [
        "F_frontier_main.csv", "C_frontier.csv", "C8_task_frontier.csv", "C6_bridge_fit.csv",
        "mechanism_series.csv", "forecast_scenarios.csv", "forecast_intervals.csv",
        "bridge_penalty_cv.csv", "contribution_identified_set_bootstrap.csv",
        "ability_metric_validation.json", "family_sensitivity_summary.json",
        "bridge_model_comparison.csv", "bridge_pythia_leave_one.csv",
        "forecast_baseline_comparison.csv", "forecast_scenario_matrix.csv",
        "C8_quality_audit.json", "data_contract.csv", "uncertainty_sources.csv",
        "frontier_sensitivity.csv", "rolling_origin_backtest.csv", "decomposition_summary.json",
        "forecast_summary.json", "validation_summary.json",
    ]
    missing = [name for name in required if not (OUT / name).exists() or (OUT / name).stat().st_size == 0]
    assert not missing, f"missing outputs: {missing}"

    frontier = pd.read_csv(OUT / "F_frontier_main.csv", parse_dates=["date"])
    compute = pd.read_csv(OUT / "C_frontier.csv", parse_dates=["date"])
    mechanism = pd.read_csv(OUT / "mechanism_series.csv")
    forecast = pd.read_csv(OUT / "forecast_scenarios.csv")
    intervals = pd.read_csv(OUT / "forecast_intervals.csv")
    contribution = pd.read_csv(OUT / "contribution_identified_set_bootstrap.csv")
    c8 = pd.read_csv(OUT / "C8_task_frontier.csv")
    curve = pd.read_csv(OUT / "C6_bridge_curve.csv")
    bridge_comparison = pd.read_csv(OUT / "bridge_model_comparison.csv")
    baseline_comparison = pd.read_csv(OUT / "forecast_baseline_comparison.csv")
    ability_validation = json.loads((OUT / "ability_metric_validation.json").read_text(encoding="utf-8"))

    assert frontier["date"].max() <= ANALYSIS_END
    assert compute["date"].max() <= ANALYSIS_END
    assert c8["task"].nunique() == 6
    assert np.all(np.diff(curve["benchmark"]) <= 1e-10), "bridge must be non-increasing"
    assert np.allclose(mechanism["delta_F"], mechanism["delta_B"] + mechanism["T"], atol=1e-10)
    assert np.allclose(mechanism["delta_F"], mechanism["delta_B_tail"] + mechanism["T_tail"], atol=1e-10)
    assert (contribution["scale_gain_lower"] <= contribution["scale_gain_upper"]).all()
    assert forecast["forecast"].between(0, 100).all()
    assert (intervals["lower_2_5"] <= intervals["median"]).all()
    assert (intervals["median"] <= intervals["upper_97_5"]).all()
    assert [x for x, _ in Q3_ANCHORS] == [1e19, 1e22, 1e24]
    assert ability_validation["all_three_frontiers_same_direction"]
    assert bridge_comparison["bridge"].nunique() == 4
    assert baseline_comparison["model"].nunique() == 3

    manifest = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"output_manifest_sha256.json", "verification_summary.json"}:
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "output_manifest_sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "required_outputs": len(required),
        "manifest_files": len(manifest),
        "identity_max_abs_error": float(np.max(np.abs(mechanism["delta_F"] - mechanism["delta_B"] - mechanism["T"]))),
        "analysis_end": str(ANALYSIS_END),
    }
    (OUT / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
