"""Rebuild the complete C8 per-task table from the archived or extracted records.

Strict JSON parsing only; truncated files are skipped and reported. For each
model directory the latest parseable record is retained, following the official
data guidance (1,863 dirs / 1,958 JSON / 4 truncated).

Two layouts are accepted, and they yield the same table:

* an archive (``F題.zip``) containing ``real_attachments/C_efficiency_evolution/
  detailed_results/``, or
* the already-extracted ``data/real_attachments/C_efficiency_evolution/
  detailed_results/`` tree.

Only the archive carries the four truncated records, so the summary reports which
source was used.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
ATTACHMENTS = ROOT.parents[1] / "data" / "real_attachments"
ZIP_PATH = ATTACHMENTS / "F題.zip"
DETAILED = ATTACHMENTS / "C_efficiency_evolution" / "detailed_results"
PREFIX = "real_attachments/C_efficiency_evolution/detailed_results/"

FAMILY_METRIC = {
    "IFEval": ("leaderboard_ifeval", "inst_level_strict_acc,none"),
    "BBH": ("leaderboard_bbh", "acc_norm,none"),
    "MATH_Lvl5": ("leaderboard_math_hard", "exact_match,none"),
    "GPQA": ("leaderboard_gpqa", "acc_norm,none"),
    "MUSR": ("leaderboard_musr", "acc_norm,none"),
    "MMLU_PRO": ("leaderboard_mmlu_pro", "acc,none"),
}


def record_timestamp(name: str) -> str:
    m = re.search(r"results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", name)
    return m.group(1) if m else name


def read_archive() -> list[tuple[str, str, bytes]]:
    """``(model, full name, payload)`` for every JSON in the archive."""

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = [n for n in archive.namelist() if n.startswith(PREFIX) and n.lower().endswith(".json")]
        return [(name[len(PREFIX):].split("/")[0], name, archive.read(name)) for name in names]


def read_extracted() -> list[tuple[str, str, bytes]]:
    """Same triples, read from the directory tree instead of the archive."""

    records = []
    for path in sorted(DETAILED.glob("*/*.json")):
        model = path.parent.name
        records.append((model, f"{PREFIX}{model}/{path.name}", path.read_bytes()))
    return records


def load_records() -> tuple[list[tuple[str, str, bytes]], str]:
    if ZIP_PATH.is_file():
        return read_archive(), f"archive:{ZIP_PATH.name}"
    if DETAILED.is_dir():
        return read_extracted(), "extracted:detailed_results/"
    raise SystemExit(
        f"neither {ZIP_PATH} nor {DETAILED} exists; place the C-type attachment under {ATTACHMENTS}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, source = load_records()

    parsed_by_model: dict[str, list[tuple[str, str, dict]]] = {}
    truncated = []
    for model, name, payload in records:
        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            truncated.append({"file": name, "error": type(exc).__name__, "message": str(exc)[:140]})
            continue
        parsed_by_model.setdefault(model, []).append((record_timestamp(name), name, obj))

    rows = []
    for model, model_records in parsed_by_model.items():
        _, chosen_file, obj = max(model_records, key=lambda item: item[0])
        scores = {}
        for family, (key, metric) in FAMILY_METRIC.items():
            value = obj.get("results", {}).get(key, {}).get(metric)
            scores[family] = round(float(value) * 100, 4) if isinstance(value, (int, float)) else None
        rows.append({"model": model, "record_file": chosen_file,
                     "timestamp": record_timestamp(chosen_file),
                     "six_dim_complete": all(v is not None for v in scores.values()),
                     **scores})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "C8_per_task_scores.csv", index=False, encoding="utf-8-sig")

    all_models = {model for model, _, _ in records}
    lost_models = sorted(all_models - set(parsed_by_model))
    recovered = [m for m in parsed_by_model if len(parsed_by_model[m]) > 1]
    summary = {
        "source": source,
        "json_entries": len(records),
        "parsed_json": sum(len(v) for v in parsed_by_model.values()),
        "truncated_json": len(truncated),
        "truncated_files": truncated,
        "model_dirs": len(all_models),
        "model_dirs_with_parseable": len(parsed_by_model),
        "six_dim_complete_models": int(table.six_dim_complete.sum()),
        "lost_models": lost_models,
        "recovered_models_with_earlier_record": recovered,
        "metric_definition": {k: f"{key}.{metric}" for k, (key, metric) in FAMILY_METRIC.items()},
    }
    (OUT / "C8_rebuild_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C8 per-task table:", table.shape)
    print("source:", source)
    print("truncated:", len(truncated), "lost models:", lost_models, "recovered:", len(recovered))


if __name__ == "__main__":
    main()
