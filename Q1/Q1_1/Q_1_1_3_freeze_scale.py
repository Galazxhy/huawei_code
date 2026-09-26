"""Fit the shared seven-domain reference scale on A1, then freeze it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

# Direction is a documented modeling assumption, not an empirical property.
# 'middle' denotes a soft suitability band on the shared percentile scale.
DIRECTIONS = {
    "fineweb_edu": ("positive", "educational value"),
    "fluency_en": ("positive", "probability of fluent text"),
    "modernbert_cleanliness": ("positive", "cleanliness ordinal rating"),
    "modernbert_readability": ("positive", "readability ordinal rating"),
    "modernbert_reasoning": ("positive", "reasoning ordinal rating; domain suitability caveat"),
    "modernbert_professionalism": ("positive", "professionalism ordinal rating; domain suitability caveat"),
    "dsir_books": ("positive", "relative importance for book-like text; may encode domain rather than quality"),
    "dsir_wiki": ("positive", "relative importance for Wikipedia-like text; may encode domain rather than quality"),
    "dsir_math": ("positive", "relative importance for math-like text; may encode domain rather than quality"),
    "qurater_writing_style": ("positive", "writing-style rating"),
    "qurater_required_expertise": ("positive", "expertise rating is treated as higher-is-better"),
    "qurater_facts_and_trivia": ("positive", "facts/trivia rating"),
    "qurater_educational_value": ("positive", "educational-value rating"),
    "ad_en": ("positive", "probability of no advertisement"),
    "rps_doc_word_count": ("middle", "very short/inordinately long content can both be unsuitable"),
    "rps_doc_num_sentences": ("middle", "length/structure proxy, not intrinsically monotone"),
    "rps_doc_unigram_entropy": ("middle", "both repetitive and extremely diverse text can be noisy"),
    "rps_doc_frac_unique_words": ("positive", "higher lexical diversity is favorable"),
    "rps_doc_frac_no_alph_words": ("negative", "non-alphabetic noise proxy; code/math domain caveat"),
    "rps_doc_frac_chars_top_2gram": ("negative", "excessive n-gram repetition proxy"),
    "rps_doc_frac_chars_top_3gram": ("negative", "excessive n-gram repetition proxy"),
    "rps_lines_uppercase_letter_fraction": ("middle", "upper-case formatting extremes; domain dependent"),
    "rps_lines_ending_with_terminal_punctution_mark": ("positive", "well-formed prose proxy; code domain caveat"),
    "rps_lines_numerical_chars_fraction": ("positive", "numeric density retained as a favorable signal after domain audit"),
    "rps_doc_mean_word_length": ("middle", "very short/long words can both be unsuitable"),
}

QURATER_COLUMNS = [name for name in DIRECTIONS if name.startswith("qurater_")]
FINAL_FIELDS = [name for name in DIRECTIONS if not name.startswith("qurater_")] + ["qurater"]


def percentile(values: np.ndarray, domain_sorted: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(values.size, dtype=float)
    for reference in domain_sorted:
        lower = np.searchsorted(reference, values, side="left")
        upper = np.searchsorted(reference, values, side="right")
        out += (lower + upper) / (2.0 * len(reference) * len(domain_sorted))
    return out


def suitability(p: np.ndarray, direction: str) -> np.ndarray:
    if direction == "positive":
        return p
    if direction == "negative":
        return 1 - p
    # Soft central band: full score on central ranks, linear taper in both tails.
    return np.clip(np.minimum(p / 0.20, (1 - p) / 0.20), 0, 1)


def score_frame(frame: pd.DataFrame, references: dict[str, list[np.ndarray]], medians: dict[str, float], domain_medians: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    features = {}
    for field, (direction, _) in DIRECTIONS.items():
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float, copy=True)
        invalid = ~np.isfinite(values)
        if domain_medians is not None and "domain" in frame:
            for domain, domain_median in domain_medians[field].items():
                mask = invalid & (frame["domain"].to_numpy() == domain)
                values[mask] = domain_median
        values[~np.isfinite(values)] = medians[field]
        features[field] = suitability(percentile(values, references[field]), direction)
    result = pd.DataFrame({name: features[name] for name in FINAL_FIELDS if name != "qurater"})
    result["qurater"] = np.mean([features[name] for name in QURATER_COLUMNS], axis=0)
    return result[FINAL_FIELDS]


def main() -> None:
    frame = pd.read_csv(OUT / "A1_scalarized_signals.csv", low_memory=False)
    domains = sorted(frame["domain"].dropna().unique().tolist())
    if len(domains) != 7 or frame.shape[0] != 51230:
        raise ValueError(f"Unexpected A1 size/domain coverage: {frame.shape[0]}, {domains}")
    references: dict[str, list[np.ndarray]] = {}
    medians: dict[str, float] = {}
    domain_medians: dict[str, dict[str, float]] = {}
    archive: dict[str, np.ndarray] = {}
    audit_rows = []

    for field, (direction, reason) in DIRECTIONS.items():
        per_domain = []
        domain_medians[field] = {}
        for index, domain in enumerate(domains):
            group = pd.to_numeric(frame.loc[frame["domain"] == domain, field], errors="coerce").to_numpy(dtype=float, copy=True)
            finite = group[np.isfinite(group)]
            if finite.size == 0:
                raise ValueError(f"{field} has no finite observations in {domain}")
            median = float(np.median(finite))
            domain_medians[field][domain] = median
            group[~np.isfinite(group)] = median
            reference = np.sort(group)
            archive[f"{field}__{index}"] = reference
            per_domain.append(reference)
            audit_rows.append({
                "field": field, "domain": domain, "direction": direction,
                "assumption": reason, "rows": len(group),
                "invalid_count": int(len(group) - finite.size), "domain_median": median,
                "p005": float(np.quantile(reference, .005)),
                "p50": float(np.quantile(reference, .5)),
                "p995": float(np.quantile(reference, .995)),
            })
        references[field] = per_domain
        medians[field] = float(np.median(np.concatenate(per_domain)))

    scored = score_frame(frame, references, medians, domain_medians)
    if scored.isna().any().any() or ((scored < 0) | (scored > 1)).any().any():
        raise ValueError("Invalid normalized scores")
    scored.insert(0, "domain", frame["domain"])
    scored.insert(0, "id", frame["id"])
    scored.to_csv(OUT / "A1_directional_signals.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / "A1_frozen_ecdf.npz", **archive)
    pd.DataFrame(audit_rows).to_csv(OUT / "A1_direction_and_domain_audit.csv", index=False, encoding="utf-8-sig")
    parameters = {
        "domains": domains, "directions": {k: v[0] for k, v in DIRECTIONS.items()},
        "median_fallback_global": medians,
        "median_fallback_by_domain": domain_medians,
        "reference_file": "A1_frozen_ecdf.npz",
        "tie_rule": "midrank: (count_less + count_less_or_equal)/(2*n)",
        "domain_rule": "each A1 domain contributes 1/7 to the shared ECDF",
        "qurater_rule": "transform each of four semantic dimensions separately, then average to one field score",
        "middle_rule": "score=min(F/0.20,(1-F)/0.20,1); provisional semantic suitability band",
        "missing_rule": "impute A1 same-domain median during fitting; for unseen domains use frozen A1 global median",
        "reference_and_directions": "https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/raw/main/README.md",
    }
    (OUT / "A1_frozen_scale.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen seven-domain ECDFs for {len(DIRECTIONS)} transformed columns -> {len(FINAL_FIELDS)} original signals")
    print("A1 directional signals:", scored.shape, "range:", scored.iloc[:, 2:].min().min(), scored.iloc[:, 2:].max().max())


if __name__ == "__main__":
    main()
