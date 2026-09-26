"""Verify all third-subquestion deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    checks = {}
    assert pd.read_csv(OUT / "P_train_1m.csv").shape == (512, 17)
    assert pd.read_csv(OUT / "Z_train.csv").shape == (512, 13)
    assert pd.read_csv(OUT / "X2_train_1m.csv").shape == (512, 153)
    for key, shape in (("test_1m", (256, 17)), ("test_60m", (256, 17)), ("test_1b", (64, 17)),
                       ("est_10b", (63, 17)), ("est_70b", (63, 17))):
        assert pd.read_csv(OUT / f"P_{key}.csv").shape == shape, key
        assert pd.read_csv(OUT / f"Z_{key}.csv").shape[0] == shape[0], key
    B = pd.read_csv(OUT / "B_multitask_sparse_scheffe.csv", index_col=0)
    assert B.shape == (153, 13)
    assert pd.read_csv(OUT / "substitution_matrix.csv", index_col=0).shape == (17, 17)
    assert pd.read_csv(OUT / "complementarity_matrix.csv", index_col=0).shape == (17, 17)
    assert len(pd.read_csv(OUT / "validation_1m_per_task.csv")) == 13
    assert len(pd.read_csv(OUT / "cross_scale_validation.csv")) == 4
    assert len(pd.read_csv(OUT / "cross_scale_per_domain.csv")) == 4 * 14
    assert pd.read_csv(OUT / "mixture_domain_support.csv").shape == (17, 4)
    assert pd.read_csv(OUT / "pairwise_joint_effect_at_reference.csv", index_col=0).shape == (17, 17)
    assert len(pd.read_csv(OUT / "model_gap_uncertainty.csv")) == 100
    assert pd.read_csv(OUT / "substitution_eligibility.csv").shape[1] >= 5
    assert pd.read_csv(OUT / "complementarity_eligibility.csv").shape[1] >= 5
    assert len(pd.read_csv(OUT / "regmix_experiment_manifest.csv")) == 6
    assert len(pd.read_csv(OUT / "phi_scale_transfer.csv")) == 5
    assert pd.read_csv(OUT / "Q17_mapping.csv").shape == (17, 3)
    json.loads((OUT / "validation_summary.json").read_text(encoding="utf-8"))
    json.loads((OUT / "quality_prior_summary.json").read_text(encoding="utf-8"))
    checks = {"train_recipes": 512, "validation_loss_domains": 13, "mixture_domains": 17,
              "scheffe_features": 153, "test_1m": 256, "test_60m": 256, "test_1b": 64,
              "est_10b": 63, "est_70b": 63}
    (OUT / "verification_summary.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print("Verified:", checks)


if __name__ == "__main__":
    main()
