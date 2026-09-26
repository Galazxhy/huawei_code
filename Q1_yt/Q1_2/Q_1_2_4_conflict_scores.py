"""Turn cross-fitted residuals into standardized anomaly and conflict scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "results"
ALPHA = 0.025
GAMMA = 0.5
MAD_SCALE = 1.4826


def main() -> None:
    a1 = pd.read_csv(OUT / "A1_crossfit_residuals.csv")
    names = [c for c in a1.columns if c not in ("id", "domain")]
    domains = sorted(a1.domain.unique().tolist())

    scales: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        scales[name] = {}
        for domain in domains:
            e = a1.loc[a1.domain == domain, name].to_numpy(dtype=float)
            sigma = float(MAD_SCALE * np.median(np.abs(e - np.median(e))))
            r_abs = np.abs(e) / max(sigma, 1e-9)
            scales[name][domain] = {
                "sigma": sigma,
                "r_p50": float(np.quantile(r_abs, 0.50)),
                "r_threshold_alpha": float(np.quantile(r_abs, 1 - ALPHA)),
            }
            scales[name][domain]["r_threshold_alpha"] = max(scales[name][domain]["r_threshold_alpha"], 1e-6)
    (OUT / "conflict_scale.json").write_text(
        json.dumps({"alpha": ALPHA, "scales": scales, "domains": domains}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    datasets = {"A1": a1, "A2": pd.read_csv(OUT / "A2_crossfit_residuals.csv"),
                "A3": pd.read_csv(OUT / "A3_crossfit_residuals.csv")}
    for dataset_name, frame in datasets.items():
        n = len(frame)
        r_matrix = np.empty((n, len(names)), dtype=float)
        flag_matrix = np.zeros((n, len(names)), dtype=bool)
        for j, name in enumerate(names):
            e = frame[name].to_numpy(dtype=float)
            r_col = np.empty(n, dtype=float)
            for domain in domains:
                mask = frame.domain.to_numpy() == domain
                sigma = scales[name][domain]["sigma"]
                r_col[mask] = np.abs(e[mask]) / max(sigma, 1e-9)
            r_matrix[:, j] = r_col
            # Use the A1 domain empirical r distribution for the p-value (frozen model).
            for domain in domains:
                mask = frame.domain.to_numpy() == domain
                if not mask.any():
                    continue
                a1_mask = a1.domain.to_numpy() == domain
                a1_r = np.abs(a1.loc[a1_mask, name].to_numpy(dtype=float)) / max(scales[name][domain]["sigma"], 1e-9)
                sorted_r = np.sort(a1_r)
                # p_conf = P(R >= r); smaller means more anomalous.
                ranks_greater_equal = np.searchsorted(sorted_r, r_col[mask], side="left")
                p_conf = 1 - ranks_greater_equal / len(sorted_r)
                flag_matrix[mask, j] = p_conf < ALPHA

        n_conflict = flag_matrix.sum(axis=1)
        A = n_conflict / len(names)
        with np.errstate(divide="ignore", invalid="ignore"):
            B = np.where(n_conflict > 0,
                         (flag_matrix * r_matrix).sum(axis=1) / np.maximum(n_conflict, 1),
                         0.0)
        C = np.sqrt(np.maximum(A, 0) * np.maximum(B, 0))
        scores = pd.DataFrame({
            "id": frame.id, "domain": frame.domain,
            "n_conflict": n_conflict, "A_coverage": A, "B_intensity": B, "C_conflict": C,
        })
        scores.to_csv(OUT / f"{dataset_name}_conflict_scores.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(OUT / f"{dataset_name}_anomaly.npz",
                            r=r_matrix, flag=flag_matrix, names=np.array(names, dtype=object))
        print(f"{dataset_name}: conflict scores written; flagged indicator rate={float(n_conflict.mean()/len(names)):.4f}")


if __name__ == "__main__":
    main()
