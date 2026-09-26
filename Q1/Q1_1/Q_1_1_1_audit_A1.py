"""Audit the A1 quality-signal sample before choosing scoring transformations."""

from __future__ import annotations

import csv
import json
import lzma
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ATTACHMENT_DIR = PROJECT_DIR.parents[1] / "data" / "real_attachments" / "A_data_value"
OUTPUT_DIR = PROJECT_DIR / "results"

QUALITY_FIELDS = [
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

LIST_FIELDS = {
    "fineweb_edu": {0: "score"},
    "fluency_en": {0: "not_fluent_logit", 1: "fluent_logit"},
    "modernbert_cleanliness": {i: f"level_{i}_logit" for i in range(6)},
    "modernbert_readability": {i: f"level_{i}_logit" for i in range(6)},
    "modernbert_reasoning": {i: f"level_{i}_logit" for i in range(6)},
    "modernbert_professionalism": {i: f"level_{i}_logit" for i in range(6)},
    "qurater": {
        0: "writing_style",
        1: "required_expertise",
        2: "facts_and_trivia",
        3: "educational_value",
    },
    "ad_en": {0: "has_ad_logit", 1: "no_ad_logit"},
}

SOURCE_GUIDANCE = {
    "fineweb_edu": "Higher educational-value rating is favorable.",
    "fluency_en": "Binary logits [not_fluent, fluent]; convert to P(fluent).",
    "modernbert_cleanliness": "Six logits for ordinal levels 0-5; use expected level after softmax.",
    "modernbert_readability": "Six logits for ordinal levels 0-5; use expected level after softmax.",
    "modernbert_reasoning": "Six logits for ordinal levels 0-5; use expected level after softmax.",
    "modernbert_professionalism": "Six logits for ordinal levels 0-5; use expected level after softmax.",
    "dsir_books": "Importance/relevance signal relative to Books; direction and cross-domain comparability require verification.",
    "dsir_wiki": "Importance/relevance signal relative to Wikipedia; direction and cross-domain comparability require verification.",
    "dsir_math": "Importance/relevance signal relative to mathematics; direction and cross-domain comparability require verification.",
    "qurater": "Four semantically distinct scores; do not average until their scale and treatment are justified.",
    "ad_en": "Binary logits [has_ad, no_ad]; convert to P(no_ad).",
    "rps_doc_word_count": "Document length proxy; not intrinsically monotone in quality.",
    "rps_doc_num_sentences": "Document length proxy; not intrinsically monotone in quality.",
    "rps_doc_unigram_entropy": "Lexical diversity proxy; direction may depend on domain and noise.",
    "rps_doc_frac_unique_words": "Lexical diversity/degeneracy proxy; direction requires empirical and semantic review.",
    "rps_doc_frac_no_alph_words": "Higher values generally indicate noisy/non-linguistic text; provisional negative direction.",
    "rps_doc_frac_chars_top_2gram": "Repeated-character/phrase concentration proxy; provisional negative direction.",
    "rps_doc_frac_chars_top_3gram": "Repeated-character/phrase concentration proxy; provisional negative direction.",
    "rps_lines_uppercase_letter_fraction": "Formatting proxy; direction is domain-dependent and requires review.",
    "rps_lines_ending_with_terminal_punctution_mark": "Higher may indicate well-formed prose; provisional positive direction.",
    "rps_lines_numerical_chars_fraction": "Content/format proxy; direction is domain-dependent.",
    "rps_doc_mean_word_length": "Style/complexity proxy; not intrinsically monotone in quality.",
}


def locate_a1() -> Path:
    """Find the A1 quality-signal sample under ``data/real_attachments/A_data_value``.

    The attachment ships the sample twice -- plain ``.jsonl`` and ``.jsonl.xz``
    -- and either, or both, may be present after unpacking.  The plain file wins
    because it can be read without decompressing; the compressed copy is the
    fallback when only that one was kept.
    """
    plain = sorted(path for path in ATTACHMENT_DIR.rglob("slimpajama_quality_signal_sample.jsonl") if path.is_file())
    if plain:
        return plain[0]
    compressed = sorted(
        path for path in ATTACHMENT_DIR.rglob("slimpajama_quality_signal_sample.jsonl.xz") if path.is_file()
    )
    if compressed:
        return compressed[0]
    raise FileNotFoundError(
        "找不到 A1 质量信号样本：请在 data/real_attachments/A_data_value/ 下放置 "
        "slimpajama_quality_signal_sample.jsonl（或其 .xz 压缩版）。"
    )


def open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return path.open("r", encoding="utf-8")


def numeric_values(value: Any) -> list[float]:
    values = value if isinstance(value, list) else [value]
    return [float(item) for item in values if isinstance(item, (int, float)) and np.isfinite(item)]


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n_numeric": 0, "min": None, "p01": None, "p05": None, "p25": None,
                "median": None, "mean": None, "p75": None, "p95": None, "p99": None, "max": None,
                "std": None}
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n_numeric": int(array.size),
        "min": float(array.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "mean": float(array.mean()),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(array.max()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1_path = locate_a1()

    row_count = 0
    domain_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = {field: Counter() for field in QUALITY_FIELDS}
    list_lengths: dict[str, Counter[int]] = {field: Counter() for field in LIST_FIELDS}
    scalar_values: dict[str, list[float]] = {field: [] for field in QUALITY_FIELDS}
    component_values: dict[str, dict[int, list[float]]] = {
        field: defaultdict(list) for field in LIST_FIELDS
    }
    domain_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    content_lengths: list[int] = []
    ids: set[str] = set()
    duplicate_ids = 0
    missing_id = 0
    missing_domain = 0
    malformed_records = 0
    observed_keys: set[str] = set()

    with open_jsonl(a1_path) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_records += 1
                continue
            row_count += 1
            observed_keys.update(record.keys())

            identifier = record.get("id")
            if not identifier:
                missing_id += 1
            elif identifier in ids:
                duplicate_ids += 1
            else:
                ids.add(identifier)

            domain = record.get("_source_domain") or record.get("sub_path") or "(missing)"
            domain = str(domain)
            domain_counts[domain] += 1
            if domain == "(missing)":
                missing_domain += 1

            content = record.get("content")
            if isinstance(content, str):
                content_lengths.append(len(content))

            for field in QUALITY_FIELDS:
                value = record.get(field)
                if value is None:
                    missing[field] += 1
                    type_counts[field]["missing"] += 1
                    continue
                type_name = "list" if isinstance(value, list) else type(value).__name__
                type_counts[field][type_name] += 1
                values = numeric_values(value)
                if not values:
                    type_counts[field]["no_numeric_value"] += 1
                if isinstance(value, list):
                    list_lengths[field][len(value)] += 1
                    for component_idx, component in enumerate(value):
                        if isinstance(component, (int, float)) and np.isfinite(component):
                            component_values[field][component_idx].append(float(component))
                elif values:
                    scalar_values[field].extend(values)
                    domain_values[domain][field].extend(values)

    if row_count == 0:
        raise ValueError("A1 contained no valid JSON records.")

    audit_rows = []
    for field in QUALITY_FIELDS:
        n_missing = missing[field]
        row = {
            "field": field,
            "observed_types": "; ".join(f"{key}:{value}" for key, value in sorted(type_counts[field].items())),
            "missing_count": n_missing,
            "missing_rate": n_missing / row_count,
            "list_length_distribution": "; ".join(
                f"{length}:{count}" for length, count in sorted(list_lengths.get(field, {}).items())
            ),
            "source_semantics_and_direction_review": SOURCE_GUIDANCE[field],
        }
        row.update(describe(scalar_values[field]))
        audit_rows.append(row)
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "A1_indicator_audit.csv", index=False, encoding="utf-8-sig")

    component_rows = []
    for field, components in component_values.items():
        for component_idx, values in sorted(components.items()):
            row = {
                "field": field,
                "component_index": component_idx,
                "component_name_from_dataset_card": LIST_FIELDS[field].get(component_idx, f"component_{component_idx}"),
            }
            row.update(describe(values))
            component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(
        OUTPUT_DIR / "A1_list_component_audit.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        [{"domain": domain, "sample_count": count, "sample_share": count / row_count}
         for domain, count in sorted(domain_counts.items())]
    ).to_csv(OUTPUT_DIR / "A1_domain_sample_counts.csv", index=False, encoding="utf-8-sig")

    domain_missing_rows = []
    for domain in sorted(domain_counts):
        domain_missing_rows.append({
            "domain": domain,
            **{field: (domain_counts[domain] - len(domain_values[domain].get(field, []))) / domain_counts[domain]
               for field in QUALITY_FIELDS if field not in LIST_FIELDS},
        })
    pd.DataFrame(domain_missing_rows).to_csv(
        OUTPUT_DIR / "A1_scalar_missing_by_domain.csv", index=False, encoding="utf-8-sig"
    )

    content_stats = describe([float(value) for value in content_lengths])
    summary = {
        "source_file": str(a1_path),
        "source_bytes": a1_path.stat().st_size,
        "actual_format": "xz-compressed JSONL" if a1_path.suffix == ".xz" else "UTF-8 JSONL",
        "records_read": row_count,
        "malformed_records": malformed_records,
        "quality_field_count": len(QUALITY_FIELDS),
        "list_field_count": len(LIST_FIELDS),
        "observed_top_level_key_count": len(observed_keys),
        "observed_top_level_keys": sorted(observed_keys),
        "domain_count": len(domain_counts),
        "domain_sample_counts": dict(sorted(domain_counts.items())),
        "missing_id_count": missing_id,
        "duplicate_id_count": duplicate_ids,
        "missing_domain_count": missing_domain,
        "content_length_characters": content_stats,
        "dataset_card": "https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater",
        "dataset_card_readme": "https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/raw/main/README.md",
        "audit_note": "This step audits raw signals only; it does not select final monotone directions or define Q.",
    }
    (OUTPUT_DIR / "A1_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"A1 audit complete: {row_count:,} valid records, {len(domain_counts)} domains")
    print(f"Actual source: {a1_path}")
    print(f"Audit outputs: {OUTPUT_DIR}")
    print("Domain counts:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count:,}")
    print(f"Malformed JSON lines: {malformed_records}")
    print(f"Missing IDs / duplicate IDs: {missing_id} / {duplicate_ids}")


if __name__ == "__main__":
    main()
