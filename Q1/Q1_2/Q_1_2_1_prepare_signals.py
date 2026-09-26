"""Prepare 22 standardized quality signals for A1/A2/A3 from the frozen Q1.1 model."""

from __future__ import annotations

import csv
import json
import lzma
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
Q1_1 = ROOT.parent / "Q1_1"
if str(Q1_1) not in sys.path:
    sys.path.insert(0, str(Q1_1))

from Q_1_1_2_scalarize_A1 import scalarize  # noqa: E402
from Q_1_1_3_freeze_scale import DIRECTIONS, score_frame  # noqa: E402


OUT = ROOT / "results"
SOURCE = ROOT.parents[1] / "data" / "real_attachments" / "A_data_value" / "slimpajama_quality_extended"


def extension_file(domain: str) -> Path:
    """Locate one domain's extension file.

    The attachment ships each part twice (``.jsonl`` and ``.jsonl.xz``), so the
    plain file is preferred and the compressed copy is the fallback.
    """
    plain = sorted(SOURCE.glob(f"{domain}_*.jsonl"))
    candidates = plain or sorted(SOURCE.glob(f"{domain}_*.jsonl.xz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {domain} extension file under {SOURCE} "
            f"(plain .jsonl preferred, .jsonl.xz accepted), got {candidates}"
        )
    return candidates[0]


def process_extension(path: Path, domain: str, target: Path,
                      references, global_medians, domain_medians) -> None:
    batch: list[dict] = []
    rows = 0
    with target.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["id", "domain", *[name for name in DIRECTIONS if not name.startswith("qurater_")], "qurater"])

        def flush() -> None:
            nonlocal rows
            if not batch:
                return
            chunk = pd.DataFrame(batch)
            standardized = score_frame(chunk, references, global_medians, domain_medians)
            if standardized.isna().any().any() or ((standardized < 0) | (standardized > 1)).any().any():
                raise ValueError(f"Invalid standardized signals in {domain}")
            for identifier, values in zip(chunk.id, standardized.itertuples(index=False, name=None)):
                writer.writerow([identifier, domain, *[format(v, ".10g") for v in values]])
            rows += len(chunk)
            batch.clear()

        stream = lzma.open(path, "rt", encoding="utf-8") if path.suffix == ".xz" else path.open(encoding="utf-8")
        with stream:
            for line in stream:
                record = json.loads(line)
                signals, _ = scalarize(record)
                batch.append({"id": record.get("id"), "domain": domain, **signals})
                if len(batch) >= 4000:
                    flush()
            flush()
    print(f"{domain}: {rows:,} standardized signal rows -> {target.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((Q1_1 / "results" / "A1_frozen_scale.json").read_text(encoding="utf-8"))
    with np.load(Q1_1 / "results" / "A1_frozen_ecdf.npz") as archive:
        references = {field: [archive[f"{field}__{i}"].copy() for i in range(7)] for field in DIRECTIONS}
    global_medians = params["median_fallback_global"]
    domain_medians = params["median_fallback_by_domain"]

    process_extension(extension_file("arxiv"), "arxiv", OUT / "A2_directional_signals.csv",
                      references, global_medians, domain_medians)
    process_extension(extension_file("github"), "github", OUT / "A3_directional_signals.csv",
                      references, global_medians, domain_medians)

    # Reference copy of A1 signals for a self-contained conflict folder.
    a1 = pd.read_csv(Q1_1 / "results" / "A1_directional_signals.csv")
    assert a1.shape[0] == 51230 and a1.shape[1] == 24
    a1.to_csv(OUT / "A1_directional_signals.csv", index=False, encoding="utf-8-sig")
    print("A1 standardized signals copied:", a1.shape)


if __name__ == "__main__":
    main()
