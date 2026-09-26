"""Verify all second-subquestion deliverables without changing model parameters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    checks = {}
    for dataset, count in (("A1", 51230), ("A2", 17523), ("A3", 203752)):
        residuals = pd.read_csv(OUT / f"{dataset}_crossfit_residuals.csv")
        assert len(residuals) == count
        scores = pd.read_csv(OUT / f"{dataset}_conflict_scores.csv")
        assert len(scores) == count and scores.C_conflict.notna().all()
        final = pd.read_csv(OUT / f"{dataset}_final_quality.csv")
        assert len(final) == count and final.Q_star.between(0, 1).all()
        assert final.U_reliability.between(0, 1).all()
        checks[f"{dataset}_rows"] = count
    assert len(pd.read_csv(OUT / "A1_quality_dependency_edges.csv")) == 22 * 21 // 2
    assert len(pd.read_csv(OUT / "A1_conflict_scores.csv").columns) == 6
    assert len(pd.read_csv(OUT / "conflict_text_cases.csv")) == 14
    graph = json.loads((OUT / "A1_relation_graph_summary.json").read_text(encoding="utf-8"))
    assert graph["n_nodes"] == 22 and graph["n_edges"] > 0
    for name in ["fig_Q1_2_relation_graph.png", "fig_Q1_2_indicator_domain_conflict.png",
                 "fig_Q1_2_conflict_pair_heatmap.png", "fig_Q1_2_Q0_vs_Qstar.png",
                 "fig_Q1_2_baseline_conflict.png", "fig_Q1_2_sensitivity.png"]:
        assert (OUT / name).stat().st_size > 1000, name
    checks["indicators"] = 22
    checks["high_conflict_text_cases"] = 14
    (OUT / "verification_summary.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("Verified:", checks)


if __name__ == "__main__":
    main()
