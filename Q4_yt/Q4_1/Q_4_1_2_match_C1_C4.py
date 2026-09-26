"""Entity-resolve C1 leaderboard models to C4 metadata via fuzzy matching."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CDATA = ROOT.parents[1] / "F 题" / "F题" / "real_attachments" / "C_efficiency_evolution"


def tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    c1 = pd.read_csv(CDATA / "leaderboard_cleaned.csv")
    c2 = pd.read_csv(CDATA / "leaderboard_enhanced.csv")
    c4 = pd.read_csv(CDATA / "epoch_all_ai_models.csv", low_memory=False)

    c1_norm = c1["Model"].map(lambda s: "".join(tokens(str(s))))
    c4_norm = c4["Model"].map(lambda s: "".join(tokens(str(s))))
    c4["_norm"] = c4_norm
    c4["_tokens"] = c4["Model"].map(lambda s: set(tokens(str(s))))
    c4["_date"] = pd.to_datetime(c4["Publication date"], errors="coerce")

    # Candidate pool: open-weight models published before the end of C1 coverage.
    candidates = c4[(c4["Open model weights?"].astype(str).eq("Yes")) &
                    (c4["_date"] <= pd.Timestamp("2025-03-31"))].copy()
    index = defaultdict(list)
    for pos, toks in candidates["_tokens"].items():
        for t in toks:
            index[t].append(pos)

    rows = []
    exact = 0
    for model, norm in zip(c1["Model"], c1_norm):
        c1_toks = set(tokens(str(model)))
        hit_counts = defaultdict(int)
        for t in c1_toks:
            for pos in index.get(t, []):
                hit_counts[pos] += 1
        if not hit_counts:
            rows.append({"model": model, "norm": norm, "best_c4_model": None,
                         "best_score": 0.0, "match_level": "unmatched", "c4_open": None,
                         "c4_date": None, "c4_params": None, "c4_compute": None, "c4_dataset": None})
            continue
        # Candidate ranking by token overlap / union.
        scored = []
        for pos, hits in hit_counts.items():
            cand_toks = candidates.loc[pos, "_tokens"]
            jac = hits / max(1, len(c1_toks | cand_toks))
            scored.append((jac, pos))
        scored.sort(reverse=True)
        top_pos = [pos for _, pos in scored[:25]]
        best = max((char_ratio(norm, candidates.loc[pos, "_norm"]), pos) for pos in top_pos)
        score, pos = best
        cand = candidates.loc[pos]
        if score >= 0.92:
            level = "high"
        elif score >= 0.70:
            level = "medium"
        else:
            level = "low"
        rows.append({
            "model": model, "norm": norm, "best_c4_model": cand["Model"],
            "best_score": round(score, 4), "match_level": level,
            "c4_open": cand["Open model weights?"],
            "c4_date": str(cand["_date"].date()) if pd.notna(cand["_date"]) else None,
            "c4_params": cand["Parameters"], "c4_compute": cand["Training compute (FLOP)"],
            "c4_dataset": cand["Training dataset size (total)"],
        })

    match = pd.DataFrame(rows)
    match.to_csv(OUT / "C1_C4_entity_match.csv", index=False, encoding="utf-8-sig")

    # Merge the authoritative Epoch open-weight flag from C2 where available.
    reliable = match.match_level.isin(["high", "medium"])
    summary = {
        "c1_rows": int(len(c1)),
        "match_level_counts": match.match_level.value_counts(dropna=False).to_dict(),
        "matched_any": int((match.best_c4_model.notna()).sum()),
        "reliable_high_medium": int(reliable.sum()),
        "reliable_compute": int((reliable & match.c4_compute.notna()).sum()),
        "reliable_dataset": int((reliable & match.c4_dataset.notna()).sum()),
        "matched_compute": int(match.c4_compute.notna().sum()),
        "matched_dataset": int(match.c4_dataset.notna().sum()),
        "matched_params": int(match.c4_params.notna().sum()),
        "c2_open_weights_counts": c2["Epoch_AI_Open_Weights"].value_counts(dropna=False).to_dict(),
        "c2_epoch_date_nonnull": int(c2["Epoch_AI_Publication_Date"].notna().sum()),
        "note": "Open-weight candidate pool restricted to Publication date <= 2025-03-31.",
    }
    (OUT / "C1_C4_match_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
