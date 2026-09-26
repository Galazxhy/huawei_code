"""Cross-fitted domain-aware predictions and out-of-fold residuals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923
K_FOLDS = 5


def build_features(frame: pd.DataFrame, neighbor_names: list[str], domains: list[str]) -> np.ndarray:
    parts = [frame[neighbor_names].to_numpy(dtype=float)]
    for domain in domains[1:]:
        parts.append((frame["domain"].to_numpy() == domain).astype(float)[:, None])
    return np.hstack(parts)


def main() -> None:
    adjacency = pd.read_csv(OUT / "A1_quality_dependency_adjacency.csv", index_col=0)
    names = list(adjacency.columns)
    domains = sorted(pd.read_csv(OUT / "A1_directional_signals.csv", usecols=["domain"])["domain"].dropna().unique().tolist())

    a1 = pd.read_csv(OUT / "A1_directional_signals.csv")
    a2 = pd.read_csv(OUT / "A2_directional_signals.csv")
    a3 = pd.read_csv(OUT / "A3_directional_signals.csv")

    neighbors = {name: [n for n in names if n != name and adjacency.loc[name, n] == 1] for name in names}
    if any(not neighbor_names for neighbor_names in neighbors.values()):
        raise ValueError("Relation graph has isolated nodes; lower rho0 or add connectivity safeguard")

    rng = np.random.default_rng(SEED)
    results = {"A1": a1, "A2": a2, "A3": a3}
    for dataset_name, frame in results.items():
        residuals_frame = pd.DataFrame({"id": frame.id, "domain": frame.domain})
        for name in names:
            X = build_features(frame, neighbors[name], domains)
            y = frame[name].to_numpy(dtype=float)
            if dataset_name == "A1":
                pred = np.empty_like(y)
                skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
                for train_idx, test_idx in skf.split(X, frame.domain):
                    model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, tol=1e-4, random_state=SEED)
                    model.fit(X[train_idx], y[train_idx])
                    pred[test_idx] = model.predict(X[test_idx])
            else:
                model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, tol=1e-4, random_state=SEED)
                model.fit(X, y)
                pred = model.predict(X)
            residuals_frame[name] = y - pred
            print(f"  {dataset_name}: {name} residual done", flush=True)
        residuals_frame.to_csv(OUT / f"{dataset_name}_crossfit_residuals.csv", index=False, encoding="utf-8-sig")
        print(f"{dataset_name}: cross-fitted residuals written, rows={len(residuals_frame):,}", flush=True)

    summary = {
        "folds": K_FOLDS,
        "model": "ElasticNet on graph neighbors + domain indicators",
        "domains": domains,
        "cross_fitting": "stratified 5-fold, out-of-fold predictions for A1; full A1 fit applied frozen to A2/A3",
        "neighbors": {name: neighbor_names for name, neighbor_names in neighbors.items()},
    }
    (OUT / "crossfit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
