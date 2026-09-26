"""Tail evidence: conflict resolution concentrates on the high-conflict tail."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    final = pd.read_csv(OUT / "A1_final_quality.csv")
    delta = (final.Q_star - final.Q0).abs().to_numpy()
    C = final.C_conflict.to_numpy()
    rows = []
    overall = {"tail": "all", "n": int(len(final)), "median_abs_deltaQ": float(np.median(delta)),
               "mean_abs_deltaQ": float(np.mean(delta))}
    rows.append(overall)
    for q in (0.99, 0.95, 0.90):
        thr = float(np.quantile(C, q))
        tail = C >= thr
        rows.append({"tail": f"top_{1-q:.2f}", "threshold_C": thr, "n": int(tail.sum()),
                     "median_abs_deltaQ": float(np.median(delta[tail])),
                     "mean_abs_deltaQ": float(np.mean(delta[tail])),
                     "median_ratio_vs_all": float(np.median(delta[tail]) / max(np.median(delta), 1e-12)),
                     "share_of_total_abs_delta": float(delta[tail].sum() / max(delta.sum(), 1e-12))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "high_conflict_tail_evidence.csv", index=False, encoding="utf-8-sig")
    (OUT / "high_conflict_tail_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
