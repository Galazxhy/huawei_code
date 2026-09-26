"""Build first- and second-order Scheffe mixture features for all RegMix tables."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "results"


def scheffe_features(P: np.ndarray, domains: list[str]) -> tuple[np.ndarray, list[str]]:
    linear = P.copy()
    linear_names = list(domains)
    pairs = []
    pair_names = []
    for i, j in combinations(range(len(domains)), 2):
        pairs.append((P[:, i] * P[:, j])[:, None])
        pair_names.append(f"{domains[i]}:{domains[j]}")
    X = np.hstack([linear] + pairs)
    names = linear_names + pair_names
    return X, names


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    keys = ["train_1m", "test_1m", "test_60m", "test_1b", "est_10b", "est_70b"]
    for key in keys:
        P = pd.read_csv(OUT / f"P_{key}.csv").to_numpy(dtype=float)
        X, names = scheffe_features(P, domains)
        pd.DataFrame(X, columns=names).to_csv(OUT / f"X2_{key}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(P, columns=domains).to_csv(OUT / f"X1_{key}.csv", index=False, encoding="utf-8-sig")
        print(f"{key}: Scheffe X2={X.shape}, X1={P.shape}")
    pd.DataFrame({"feature": names}).to_csv(OUT / "scheffe_feature_names.csv", index=False, encoding="utf-8-sig")
    print("feature groups:", len(domains), "linear +", len(domains) * (len(domains) - 1) // 2, "pairwise")


if __name__ == "__main__":
    main()
