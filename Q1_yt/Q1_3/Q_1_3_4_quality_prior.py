"""Map 7-domain quality to 17 domains and test a quality-modulated mixture prior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
Q1_2 = ROOT.parent / "Q1_2"
SEED = 20260923

# Rule for inferred domains: content-type-based mapping to the closest 7-domain quality signal.
DOMAIN_TO_QUALITY = {
    "arxiv": "arxiv", "github": "github", "stackexchange": "stackexchange",
    "wikipedia_en": "wikipedia", "gutenberg_pg_19": "book", "pile_cc": "commoncrawl",
    "dm_mathematics": "arxiv", "freelaw": "commoncrawl", "nih_exporter": "commoncrawl",
    "pubmed_central": "arxiv", "philpapers": "arxiv", "enron_emails": "commoncrawl",
    "ubuntu_irc": "commoncrawl", "europarl": "commoncrawl", "hackernews": "commoncrawl",
    "pubmed_abstracts": "arxiv", "uspto_backgrounds": "commoncrawl",
}


def effective_mix(P: np.ndarray, q17: np.ndarray, gamma: float) -> np.ndarray:
    q_center = q17 - q17.mean()
    weights = P * np.exp(gamma * q_center)
    return weights / weights.sum(axis=1, keepdims=True)


def scheffe(P: np.ndarray, domains: list[str]) -> tuple[np.ndarray, list[str]]:
    pairs = []
    names = list(domains)
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            pairs.append((P[:, i] * P[:, j])[:, None])
            names.append(f"{domains[i]}:{domains[j]}")
    return np.hstack([P] + pairs), names


def main() -> None:
    domains = pd.read_csv(OUT / "train_domain_names.csv")["domain"].tolist()
    P_train = pd.read_csv(OUT / "P_train_1m.csv").to_numpy(dtype=float)
    P_test = pd.read_csv(OUT / "P_test_1m.csv").to_numpy(dtype=float)
    Z_train = pd.read_csv(OUT / "Z_train.csv").to_numpy(dtype=float)
    Z_test = pd.read_csv(OUT / "Z_test_1m.csv").to_numpy(dtype=float)
    fit_info = json.loads((OUT / "scheffe_fit_summary.json").read_text(encoding="utf-8"))
    alpha = float(fit_info["best_multitask"]["alpha"])
    l1_ratio = float(fit_info["best_multitask"]["l1_ratio"])

    q7 = pd.read_csv(Q1_2 / "results" / "Qstar_domain_summary.csv")
    q7 = q7[q7.dataset == "A1"].set_index("domain")["reliability_weighted_Qstar"]
    q17 = np.array([q7[DOMAIN_TO_QUALITY[d]] for d in domains], dtype=float)
    pd.DataFrame({"mixture_domain": domains, "quality_domain": [DOMAIN_TO_QUALITY[d] for d in domains],
                  "Q17": q17}).to_csv(OUT / "Q17_mapping.csv", index=False, encoding="utf-8-sig")

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gamma_grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    rows = []
    for gamma in gamma_grid:
        fold_rmse = []
        for tr, va in kf.split(P_train):
            Pt = effective_mix(P_train[tr], q17, gamma)
            Xt, _ = scheffe(Pt, domains)
            Pv = effective_mix(P_train[va], q17, gamma)
            Xv, _ = scheffe(Pv, domains)
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                        tol=1e-5, fit_intercept=False, random_state=SEED)
            model.fit(Xt, Z_train[tr])
            pred = model.predict(Xv)
            fold_rmse.append(float(np.mean(np.sqrt(np.mean((Z_train[va] - pred) ** 2, axis=0)))))
        rows.append({"gamma": gamma, "cv_macro_rmse": float(np.mean(fold_rmse))})
    gamma_table = pd.DataFrame(rows)
    gamma_table.to_csv(OUT / "quality_prior_gamma_cv.csv", index=False, encoding="utf-8-sig")
    best_gamma = float(gamma_table.loc[gamma_table.cv_macro_rmse.idxmin(), "gamma"])

    # Test performance for M0 (gamma=0) and M_Q (best gamma).
    results = {}
    for label, gamma in (("M0_pure_mix", 0.0), ("MQ_quality_prior", best_gamma)):
        Pt = effective_mix(P_train, q17, gamma)
        Xt, _ = scheffe(Pt, domains)
        Pv = effective_mix(P_test, q17, gamma)
        Xv, _ = scheffe(Pv, domains)
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                    tol=1e-5, fit_intercept=False, random_state=SEED)
        model.fit(Xt, Z_train)
        pred = model.predict(Xv)
        rmse = float(np.mean(np.sqrt(np.mean((Z_test - pred) ** 2, axis=0))))
        mae = float(np.mean(np.abs(Z_test - pred)))
        ss_res = np.sum((Z_test - pred) ** 2, axis=0)
        ss_tot = np.sum((Z_test - Z_test.mean(axis=0)) ** 2, axis=0)
        r2 = float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-9)))
        results[label] = {"gamma": gamma, "test_macro_rmse": rmse, "test_mae": mae, "test_macro_r2": r2}
        pd.DataFrame(pred, columns=pd.read_csv(OUT / "loss_domain_names.csv")["loss_domain"]).to_csv(
            OUT / f"pred_test_1m_{label}.csv", index=False, encoding="utf-8-sig")

    summary = {"gamma_grid": rows, "best_gamma": best_gamma, "test": results,
               "mapping": DOMAIN_TO_QUALITY,
               "note": "gamma>0 means higher-quality domains get larger effective share."}
    (OUT / "quality_prior_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(gamma_table.to_string(index=False))
    print("best gamma", best_gamma)
    print(results)


if __name__ == "__main__":
    main()
