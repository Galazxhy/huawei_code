"""Bootstrap uncertainty for substitution and complementarity effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923
B_BOOT = 120


def scheffe(P: np.ndarray, domains: list[str]) -> np.ndarray:
    parts = [P]
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            parts.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(parts)


def coefficients_to_effects(B: np.ndarray, domains: list[str], p_ref: np.ndarray):
    beta = B[: len(domains)]
    pair_names = [name for name in pd.read_csv(OUT / "scheffe_feature_names.csv")["feature"] if ":" in name]
    gamma = {}
    for name in pair_names:
        i, j = name.split(":")
        gamma[frozenset((i, j))] = B[len(domains) + pair_names.index(name)]
    deriv = np.zeros((len(domains), B.shape[1]))
    for i in range(len(domains)):
        d = beta[i].copy()
        for j in range(len(domains)):
            if i == j:
                continue
            key = frozenset((domains[i], domains[j]))
            d += gamma[key] * p_ref[j]
        deriv[i] = d
    S = np.zeros((len(domains), len(domains)))
    C = np.zeros((len(domains), len(domains)))
    for i in range(len(domains)):
        for j in range(len(domains)):
            S[i, j] = (deriv[i] - deriv[j]).mean()
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            key = frozenset((domains[i], domains[j]))
            c = -gamma[key].mean()
            C[i, j] = C[j, i] = c
    return S, C


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    X = pd.read_csv(OUT / "X2_train_1m.csv").to_numpy(dtype=float)
    Y = pd.read_csv(OUT / "Z_train.csv").to_numpy(dtype=float)
    p_ref = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float).mean(axis=0)
    info = json.loads((OUT / "scheffe_fit_summary.json").read_text(encoding="utf-8"))
    alpha = float(info["best_multitask"]["alpha"])
    l1_ratio = float(info["best_multitask"]["l1_ratio"])
    rng = np.random.default_rng(SEED)

    S_all = np.zeros((B_BOOT, len(domains), len(domains)))
    C_all = np.zeros((B_BOOT, len(domains), len(domains)))
    for b in range(B_BOOT):
        idx = rng.integers(0, len(X), size=len(X))
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                    tol=1e-5, fit_intercept=False, random_state=SEED + b)
        model.fit(X[idx], Y[idx])
        S, C = coefficients_to_effects(model.coef_.T, domains, p_ref)
        S_all[b], C_all[b] = S, C

    S_mean = S_all.mean(axis=0)
    S_lo = np.quantile(S_all, 0.025, axis=0)
    S_hi = np.quantile(S_all, 0.975, axis=0)
    C_mean = C_all.mean(axis=0)
    C_lo = np.quantile(C_all, 0.025, axis=0)
    C_hi = np.quantile(C_all, 0.975, axis=0)

    S_rows, C_rows = [], []
    for i in range(len(domains)):
        for j in range(len(domains)):
            S_rows.append({"from": domains[j], "to": domains[i],
                           "S_mean": float(S_mean[i, j]), "S_lo": float(S_lo[i, j]),
                           "S_hi": float(S_hi[i, j]),
                           "sign_stable_share": float(np.mean(np.sign(S_all[:, i, j]) == np.sign(S_mean[i, j])))})
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            C_rows.append({"domain_i": domains[i], "domain_j": domains[j],
                           "C_mean": float(C_mean[i, j]), "C_lo": float(C_lo[i, j]),
                           "C_hi": float(C_hi[i, j]),
                           "sign_stable_share": float(np.mean(np.sign(C_all[:, i, j]) == np.sign(C_mean[i, j])))})
    pd.DataFrame(S_rows).to_csv(OUT / "substitution_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(C_rows).to_csv(OUT / "complementarity_bootstrap.csv", index=False, encoding="utf-8-sig")

    summary = {
        "bootstrap_replicates": B_BOOT,
        "substitution_sign_stable_share_median": float(np.median([r["sign_stable_share"] for r in S_rows])),
        "complementarity_sign_stable_share_median": float(np.median([r["sign_stable_share"] for r in C_rows])),
    }
    (OUT / "bootstrap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
