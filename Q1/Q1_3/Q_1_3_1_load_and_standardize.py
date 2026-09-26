"""Load RegMix tables and standardize the 13 validation losses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATA = ROOT.parents[1] / "data" / "real_attachments" / "A_data_value" / "regmix_tables"
MAD_SCALE = 1.4826

PAIRS = {
    "train_1m": ("train_mixture_1m.csv", "train_pile_loss_1m.csv"),
    "test_1m": ("test_mixture_1m.csv", "test_pile_loss_1m.csv"),
    "test_60m": ("test_mixture_60m.csv", "test_pile_loss_60m.csv"),
    "test_1b": ("test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    "est_10b": ("est_mixture_10b.csv", "est_pile_loss_10b.csv"),
    "est_70b": ("est_mixture_70b.csv", "est_pile_loss_70b.csv"),
}


def load_pair(mix_name: str, loss_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    mix = pd.read_csv(DATA / mix_name)
    loss = pd.read_csv(DATA / loss_name)
    mix = mix.rename(columns={mix.columns[0]: "index"})
    loss = loss.rename(columns={loss.columns[0]: "index"})
    mix = mix.sort_values("index").reset_index(drop=True)
    loss = loss.sort_values("index").reset_index(drop=True)
    if len(mix) != len(loss) or not np.array_equal(mix["index"].to_numpy(), loss["index"].to_numpy()):
        raise ValueError(f"Index mismatch in {mix_name}/{loss_name}")
    domain_names = [c.replace("train_the_pile_", "") for c in mix.columns if c.startswith("train_the_pile_")]
    loss_names = [c.replace("metric/the_pile_", "").replace("_val_loss", "")
                  for c in loss.columns if c.startswith("metric/the_pile_") and c.endswith("_val_loss")]
    P = mix[[f"train_the_pile_{d}" for d in domain_names]].to_numpy(dtype=float)
    L = loss[[f"metric/the_pile_{d}_val_loss" for d in loss_names]].to_numpy(dtype=float)
    return mix, loss, domain_names, loss_names, P, L


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, train_domains, loss_domains, P_train, L_train = load_pair(*PAIRS["train_1m"])
    print(f"Train: P={P_train.shape}, L={L_train.shape}, 17 domains={len(train_domains)}, 13 loss domains={len(loss_domains)}")

    medians = np.median(L_train, axis=0)
    mads = np.median(np.abs(L_train - medians), axis=0)
    scale = {loss_domains[k]: {"median": float(medians[k]), "mad": float(mads[k])}
             for k in range(len(loss_domains))}
    Z_train = (L_train - medians) / np.maximum(MAD_SCALE * mads, 1e-9)
    pd.DataFrame({"domain": train_domains}).to_csv(OUT / "train_domain_names.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"loss_domain": loss_domains}).to_csv(OUT / "loss_domain_names.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(P_train, columns=train_domains).to_csv(OUT / "P_train_1m.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(Z_train, columns=loss_domains).to_csv(OUT / "Z_train.csv", index=False, encoding="utf-8-sig")
    (OUT / "loss_standardization.json").write_text(json.dumps({"loss_domains": loss_domains, "scale": scale}, indent=2) + "\n", encoding="utf-8")

    for key in ("test_1m", "test_60m", "test_1b", "est_10b", "est_70b"):
        _, _, d_names, l_names, P, L = load_pair(*PAIRS[key])
        if d_names != train_domains or l_names != loss_domains:
            raise ValueError(f"Column mismatch in {key}")
        Z = (L - medians) / np.maximum(MAD_SCALE * mads, 1e-9)
        pd.DataFrame(P, columns=train_domains).to_csv(OUT / f"P_{key}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(Z, columns=loss_domains).to_csv(OUT / f"Z_{key}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(L, columns=loss_domains).to_csv(OUT / f"L_{key}.csv", index=False, encoding="utf-8-sig")
        print(f"{key}: P={P.shape}, L={L.shape}")


if __name__ == "__main__":
    main()
