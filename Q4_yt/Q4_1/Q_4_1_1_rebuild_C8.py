"""Rebuild the complete C8 per-task table from the original zip archive.

Strict JSON parsing only; truncated files are skipped and reported. For each
model directory the latest parseable record is retained, following the official
data guidance (1,863 dirs / 1,958 JSON / 4 truncated).
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
ZIP_PATH = ROOT.parents[1] / "F 题" / "F题.zip"
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(ZIP_PATH)
    entries = [n for n in archive.namelist() if n.startswith(PREFIX) and n.lower().endswith(".json")]

    parsed_by_model: dict[str, list[tuple[str, str, dict]]] = {}
    truncated = []
    for name in entries:
        model = name[len(PREFIX):].split("/")[0]
        try:
            obj = json.loads(archive.read(name).decode("utf-8"))
        except Exception as exc:
            truncated.append({"file": name, "error": type(exc).__name__, "message": str(exc)[:140]})
            continue
        parsed_by_model.setdefault(model, []).append((record_timestamp(name), name, obj))

    rows = []
    for model, records in parsed_by_model.items():
        _, chosen_file, obj = max(records, key=lambda item: item[0])
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

    all_models = {n[len(PREFIX):].split("/")[0] for n in entries}
    archive.close()
    lost_models = sorted(all_models - set(parsed_by_model))
    recovered = [m for m in parsed_by_model if len(parsed_by_model[m]) > 1]
    summary = {
        "json_entries": len(entries),
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
    (OUT / "C8_rebuild_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("C8 per-task table:", table.shape)
    print("truncated:", len(truncated), "lost models:", lost_models, "recovered:", len(recovered))


if __name__ == "__main__":
    main()
