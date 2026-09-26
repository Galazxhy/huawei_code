"""Audit C5/C6 Loss-Benchmark bridge and propose a comparability-weighted plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CDATA = ROOT.parents[1] / "F 题" / "F题" / "real_attachments" / "C_efficiency_evolution"


def main() -> None:
    c5 = pd.read_csv(CDATA / "loss_benchmark_bridge.csv")
    c6 = pd.read_csv(CDATA / "loss_benchmark_bridge_expanded.csv")

    rows = []
    for name, df in (("C5", c5), ("C6", c6)):
        for comp, group in df.groupby("Loss_Comparability", sort=True):
            rho = spearmanr(group["Val_Loss"], group["LB_Average"]).statistic
            rows.append({
                "bridge": name, "comparability": comp, "n": len(group),
                "val_loss_min": group.Val_Loss.min(), "val_loss_max": group.Val_Loss.max(),
                "lb_average_min": group.LB_Average.min(), "lb_average_max": group.LB_Average.max(),
                "D_tokens_missing": int(group.D_tokens_B.isna().sum()),
                "spearman_loss_vs_benchmark": float(rho),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "C6_bridge_stratified_summary.csv", index=False, encoding="utf-8-sig")

    # Proposed weighting rule: high comparability anchor, medium as external sensitivity.
    high = c6[c6.Loss_Comparability.str.contains("High", na=False)]
    medium = c6[c6.Loss_Comparability.str.contains("Medium", na=False)]
    summary = {
        "high_n": int(len(high)),
        "medium_n": int(len(medium)),
        "high_loss_range": [float(high.Val_Loss.min()), float(high.Val_Loss.max())],
        "high_benchmark_range": [float(high.LB_Average.min()), float(high.LB_Average.max())],
        "high_spearman": float(spearmanr(high.Val_Loss, high.LB_Average).statistic),
        "medium_spearman": float(spearmanr(medium.Val_Loss, medium.LB_Average).statistic),
        "recommendation": "High comparability anchors the monotone map; Medium comparability is weighted external sensitivity. "
                          "A high-only fit has only 7 points and is not sufficient alone.",
    }
    (OUT / "C6_bridge_plan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
