"""Package the Q1 -> Q2 handoff interface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
Q1_2 = ROOT.parent / "Q1_2"


def main() -> None:
    q7 = pd.read_csv(Q1_2 / "results" / "Qstar_domain_summary.csv")
    q7 = q7[q7.dataset == "A1"].set_index("domain")["reliability_weighted_Qstar"].to_dict()
    q17 = pd.read_csv(OUT / "Q17_mapping.csv").set_index("mixture_domain")[["quality_domain", "Q17"]].to_dict("index")
    scale = json.loads((OUT / "loss_standardization.json").read_text(encoding="utf-8"))
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    loss_domains = pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"].tolist()
    def clean_records(frame: pd.DataFrame) -> list[dict]:
        return frame.astype(object).where(pd.notna(frame), None).to_dict("records")

    experiment_manifest = clean_records(pd.read_csv(OUT / "regmix_experiment_manifest.csv"))
    phi_scale_transfer = clean_records(pd.read_csv(OUT / "phi_scale_transfer.csv"))

    handoff = {
        "Q7_final_quality": {str(k): float(v) for k, v in q7.items()},
        "Q17_final_quality": {str(k): {"quality_domain": v["quality_domain"], "Q17": float(v["Q17"])} for k, v in q17.items()},
        "mixture_domains": domains,
        "loss_domains": loss_domains,
        "loss_standardization": scale,
        "phi_definition": "phi(p) = -G(p), G(p) = (1/13) sum_k z_hat_k(p)",
        "z_to_raw_loss": "L_k = median_k + 1.4826 * MAD_k * z_k",
        "scheffe_B_file": "B_multitask_sparse_scheffe.csv",
        "quality_prior_gamma": 0.0,
        "regmix_experiment_manifest": experiment_manifest,
        "phi_scale_transfer": phi_scale_transfer,
        "D_scaling_note": "Use phi_1M(p) and phi_scale_transfer only for observed 1M/60M/1B scales; 10B/70B are extrapolated and should not calibrate absolute Q2 scaling.",
        "note": "First-question mixture data cannot identify quality's independent marginal effect; use B6-B8 in Q2.",
    }
    (OUT / "Q1_to_Q2_inputs.json").write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Handoff written:", OUT / "Q1_to_Q2_inputs.json")


if __name__ == "__main__":
    main()
