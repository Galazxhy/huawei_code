"""Check generated deliverables without changing model parameters."""

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    counts = {"A1": 51230, "A2": 17523, "A3": 203752}
    checks = {}
    for name, count in counts.items():
        frame = pd.read_csv(OUT / f"{name}_sample_Q0.csv")
        assert len(frame) == count, (name, len(frame))
        assert frame.id.notna().all() and frame.id.is_unique, name
        assert frame.Q0.notna().all() and frame.Q0.between(0, 1).all(), name
        checks[f"{name}_rows"] = count
    assignments = pd.read_csv(OUT / "A1_indicator_clusters.csv")
    assert len(assignments) == 22 and assignments.indicator.is_unique
    structure = json.loads((OUT / "A1_structure_summary.json").read_text(encoding="utf-8"))
    assert assignments.cluster.nunique() == structure["selected_k"]
    checks["indicators"] = 22
    checks["clusters"] = structure["selected_k"]
    audit = pd.read_csv(OUT / "AI_assisted_text_audit_cases.csv")
    assert audit.domain.nunique() == 7 and len(audit) == 16
    blind = pd.read_csv(OUT / "A1_manual_review_BLIND.csv", keep_default_na=False)
    assert len(blind) == 210
    assert all(blind[c].astype(str).str.strip().eq("").all() for c in
               ["readability_1to5", "information_1to5", "absence_of_noise_1to5", "completeness_1to5"])
    checks["unscored_blind_texts"] = 210
    indicator_audit = pd.read_csv(OUT / "A1_standardized_indicator_audit.csv")
    assert len(indicator_audit) == 22 and indicator_audit["原始指标"].is_unique
    checks["standardized_indicator_audit"] = 22
    ai_review = pd.read_csv(OUT / "A1_AI_review_scores.csv")
    assert len(ai_review) == 210 and ai_review.review_id.is_unique
    assert ai_review[["readability_1to5", "information_1to5", "absence_of_noise_1to5", "completeness_1to5"]].to_numpy().min() >= 1
    assert ai_review[["readability_1to5", "information_1to5", "absence_of_noise_1to5", "completeness_1to5"]].to_numpy().max() <= 5
    checks["AI_assisted_review_cases"] = 210
    extensions = pd.read_csv(OUT / "extension_representativeness.csv").set_index("domain")
    assert extensions.loc["arxiv", "id_overlap_count"] == 1419
    assert extensions.loc["github", "id_overlap_count"] == 10000
    for name in ["fig_A1_domain_counts.png", "fig_A1_clustered_correlations.png",
                 "fig_A1_domain_variance.png", "fig_A1_domain_Q0.png",
                 "fig_A1_vs_A2_A3_Q0.png", "fig_A1_model_rank_agreement.png"]:
        assert (OUT / name).stat().st_size > 1000, name
    (OUT / "verification_summary.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("Verified:", checks)


if __name__ == "__main__":
    main()
