"""Convert the heterogeneous list-valued A1 signals into auditable scalars."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
from scipy.special import softmax

from Q_1_1_1_audit_A1 import ATTACHMENT_DIR, LIST_FIELDS, QUALITY_FIELDS, locate_a1, open_jsonl


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results"
MODEL_ORDINAL_FIELDS = [
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
]
QURATER_COMPONENTS = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]


def finite_vector(value: Any, expected_size: int) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != expected_size:
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(vector).all():
        return None
    return vector


def scalarize(record: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    diagnostics: dict[str, str] = {}

    fineweb = finite_vector(record.get("fineweb_edu"), 1)
    values["fineweb_edu"] = float(fineweb[0]) if fineweb is not None else np.nan
    diagnostics["fineweb_edu"] = "singleton list; extract the sole score"

    fluency = finite_vector(record.get("fluency_en"), 2)
    values["fluency_en"] = float(softmax(fluency)[1]) if fluency is not None else np.nan
    diagnostics["fluency_en"] = "two logits [not_fluent, fluent]; use softmax probability of fluent"

    for field in MODEL_ORDINAL_FIELDS:
        logits = finite_vector(record.get(field), 6)
        values[field] = float(softmax(logits) @ np.arange(6) / 5.0) if logits is not None else np.nan
        diagnostics[field] = "six ordinal logits for levels 0-5; use normalized expected ordinal rating"

    qurater = finite_vector(record.get("qurater"), 4)
    if qurater is None:
        for component in QURATER_COMPONENTS:
            values[f"qurater_{component}"] = np.nan
    else:
        for index, component in enumerate(QURATER_COMPONENTS):
            values[f"qurater_{component}"] = float(qurater[index])
    diagnostics["qurater"] = "retain four semantically distinct rating outputs for separate scaling before field-level aggregation"

    ad = finite_vector(record.get("ad_en"), 2)
    values["ad_en"] = float(softmax(ad)[1]) if ad is not None else np.nan
    diagnostics["ad_en"] = "two logits [has_ad, no_ad]; use softmax probability of no advertisement"

    for field in QUALITY_FIELDS:
        value = record.get(field)
        if field in LIST_FIELDS:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = np.nan
        values[field] = numeric if np.isfinite(numeric) else np.nan
        diagnostics[field] = "retain numeric scalar; transform only after direction and distribution audit"

    return values, diagnostics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = locate_a1()
    rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}

    with open_jsonl(source) as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            values, _ = scalarize(record)
            row = {
                "id": record.get("id"),
                "domain": record.get("_source_domain") or record.get("sub_path"),
                "content_length": len(record.get("content", "")) if isinstance(record.get("content"), str) else np.nan,
                **values,
            }
            for field, value in values.items():
                if not np.isfinite(value):
                    failure_counts[field] = failure_counts.get(field, 0) + 1
            rows.append(row)

    frame = pd.DataFrame(rows)
    expected_features = [
        "id", "domain", "content_length", "fineweb_edu", "fluency_en",
        "modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning",
        "modernbert_professionalism", "qurater_writing_style", "qurater_required_expertise",
        "qurater_facts_and_trivia", "qurater_educational_value", "ad_en",
        *[field for field in QUALITY_FIELDS if field not in LIST_FIELDS],
    ]
    frame = frame[expected_features]
    frame.to_csv(OUTPUT_DIR / "A1_scalarized_signals.csv", index=False, encoding="utf-8-sig")

    conversion_rows = [
        {"source_field": field, "scalarization": description}
        for field, description in [
            ("fineweb_edu", "extract singleton rating"),
            ("fluency_en", "softmax(logits)[fluent]"),
            ("modernbert_cleanliness", "sum(level * softmax(logits)[level]) / 5"),
            ("modernbert_readability", "sum(level * softmax(logits)[level]) / 5"),
            ("modernbert_reasoning", "sum(level * softmax(logits)[level]) / 5"),
            ("modernbert_professionalism", "sum(level * softmax(logits)[level]) / 5"),
            ("qurater", "retain writing style, required expertise, facts/trivia, and educational value separately until each is put on a common percentile scale"),
            ("ad_en", "softmax(logits)[no_ad]"),
        ]
    ]
    pd.DataFrame(conversion_rows).to_csv(
        OUTPUT_DIR / "A1_list_scalarization_register.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "source_file": str(source),
        "rows": len(frame),
        "top_level_quality_fields": 22,
        "scalarized_columns_including_qurater_components": len(expected_features) - 3,
        "list_fields": 8,
        "missing_or_invalid_scalar_counts": failure_counts,
        "domain_counts": {str(key): int(value) for key, value in frame["domain"].value_counts().sort_index().items()},
        "transform_notes": {
            "ordinal_logits": "expected ordinal score rather than argmax to retain uncertainty",
            "binary_logits": "softmax probability for the favorable class",
            "qurater": "four dimensions are not averaged on their unverified raw scales; each is scaled separately in the next step",
            "labels": "directions are not learned from these transforms; direction register is explicit and separately auditable",
        },
    }
    (OUTPUT_DIR / "A1_scalarization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Scalarized {len(frame):,} A1 records into {len(expected_features) - 3} numeric signal columns.")
    print("List-valued signals have been converted according to dataset-card semantics.")
    print(f"Invalid or missing transformed values: {failure_counts or 'none'}")
    print(f"Output: {OUTPUT_DIR / 'A1_scalarized_signals.csv'}")


if __name__ == "__main__":
    main()
