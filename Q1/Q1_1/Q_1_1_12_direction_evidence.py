"""Direction-scheme evidence: keep only semantically supported center indicators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
Q1_1 = ROOT
if str(Q1_1) not in sys.path:
    sys.path.insert(0, str(Q1_1))

from Q_1_1_3_freeze_scale import DIRECTIONS, percentile  # noqa: E402
from Q_1_1_5_score_A1 import score_clusters  # noqa: E402


FINAL_FIELDS = [name for name in DIRECTIONS if not name.startswith("qurater_")] + ["qurater"]
QURATER_COLUMNS = [name for name in DIRECTIONS if name.startswith("qurater_")]


def suitability(p: np.ndarray, direction: str, band: float) -> np.ndarray:
    if direction == "positive":
        return p
    if direction == "negative":
        return 1 - p
    return np.clip(np.minimum(p / band, (1 - p) / band), 0, 1)


def transform(frame, references, medians, domain_medians, directions, band):
    features = {}
    for field, (direction, _) in directions.items():
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float, copy=True)
        invalid = ~np.isfinite(values)
        if domain_medians is not None and "domain" in frame:
            for domain, domain_median in domain_medians[field].items():
                mask = invalid & (frame["domain"].to_numpy() == domain)
                values[mask] = domain_median
        values[~np.isfinite(values)] = medians[field]
        features[field] = suitability(percentile(values, references[field]), direction, band)
    result = pd.DataFrame({name: features[name] for name in FINAL_FIELDS if name != "qurater"})
    result["qurater"] = np.mean([features[name] for name in QURATER_COLUMNS], axis=0)
    return result[FINAL_FIELDS]


def main() -> None:
    frame = pd.read_csv(OUT / "A1_scalarized_signals.csv")
    params = json.loads((OUT / "A1_frozen_scale.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((OUT / "A1_frozen_hierarchy.json").read_text(encoding="utf-8"))
    group_weights = {int(k): v for k, v in hierarchy["group_weights"].items()}
    with np.load(OUT / "A1_frozen_ecdf.npz") as archive:
        references = {field: [archive[f"{field}__{i}"].copy() for i in range(7)] for field in DIRECTIONS}

    baseline_q0 = pd.read_csv(OUT / "A1_sample_Q0.csv", usecols=["id", "domain", "Q0"])
    # AI-assisted rubric criterion on the 210 selected texts.
    key = pd.read_csv(OUT / "A1_manual_review_SEALED_KEY.csv")
    ai = pd.read_csv(OUT / "A1_AI_review_scores.csv")
    ai_crit = ai.merge(key[["review_id", "source_id"]], on="review_id")[["source_id", "AI_mean"]]
    ai_crit = ai_crit.merge(baseline_q0[["id", "domain", "Q0"]], left_on="source_id", right_on="id")

    # Schemes: only fields with clear semantic support remain center.
    scheme_defs = {
        "A_current_center8": {"center": {"qurater_required_expertise", "rps_doc_word_count", "rps_doc_num_sentences",
                                          "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
                                          "rps_lines_uppercase_letter_fraction", "rps_lines_numerical_chars_fraction",
                                          "rps_doc_mean_word_length"}},
        "B_center7_expertise_positive": {"center": {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
                                                    "rps_doc_frac_unique_words", "rps_lines_uppercase_letter_fraction",
                                                    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"}},
        "C_center5_structure_only": {"center": {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
                                                "rps_lines_uppercase_letter_fraction", "rps_doc_mean_word_length"}},
        "D_center4_minimal": {"center": {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
                                         "rps_doc_mean_word_length"}},
        "F_center6_with_numerical": {"center": {"rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
                                                "rps_lines_uppercase_letter_fraction", "rps_lines_numerical_chars_fraction",
                                                "rps_doc_mean_word_length"}},
        "E_all_positive": {"center": set()},
    }
    rows = []
    scheme_q0 = {}
    for scheme, definition in scheme_defs.items():
        directions = {}
        for field, (direction, reason) in DIRECTIONS.items():
            if field in definition["center"]:
                directions[field] = ("middle", reason)
            elif direction == "negative":
                directions[field] = ("negative", reason)
            else:
                directions[field] = ("positive", reason)
        signals = transform(frame, references, params["median_fallback_global"],
                            params["median_fallback_by_domain"], directions, 0.20)
        scored = score_clusters(signals, group_weights)
        scored["id"] = frame["id"].to_numpy()
        scored["domain"] = frame["domain"].to_numpy()
        scheme_q0[scheme] = scored
        m = scored.merge(baseline_q0[["id", "Q0"]], on="id", suffixes=("", "_base"))
        sample_sp = float(spearmanr(m.Q0, m.Q0_base).statistic)
        domain_sp = float(spearmanr(m.groupby("domain").Q0.mean(), m.groupby("domain").Q0_base.mean()).statistic)
        mcrit = scored.merge(ai_crit[["source_id", "AI_mean"]], left_on="id", right_on="source_id")
        ai_sp = float(spearmanr(mcrit.Q0, mcrit.AI_mean).statistic)
        rows.append({"scheme": scheme, "n_center_fields": len(definition["center"]),
                     "sample_spearman_to_baseline": sample_sp,
                     "domain_rank_spearman_to_baseline": domain_sp,
                     "spearman_with_AI_rubric_210": ai_sp})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "direction_scheme_evidence.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
