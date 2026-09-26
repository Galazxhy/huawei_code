"""Score every A2/A3 record with frozen A1 parameters; audit overlap."""

from __future__ import annotations

import csv
import json
import lzma
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance

from Q_1_1_2_scalarize_A1 import scalarize
from Q_1_1_3_freeze_scale import DIRECTIONS, score_frame
from Q_1_1_5_score_A1 import score_clusters


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCE = ROOT.parents[1] / "F 题" / "F题" / "real_attachments" / "A_data_value" / "slimpajama_quality_extended"
SEED = 20260923


def source_files() -> dict[str, Path]:
    result = {}
    for domain in ("arxiv", "github"):
        candidates = list(SOURCE.glob(f"{domain}_*.jsonl")) + list(SOURCE.glob(f"{domain}_*.jsonl.xz"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one {domain} extension file, got {candidates}")
        result[domain] = candidates[0]
    return result


def compare(domain: str, baseline: np.ndarray, extension: np.ndarray, nonoverlap: np.ndarray, overlap: int) -> dict[str, float | str | int]:
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(500):
        a = baseline[rng.integers(0, len(baseline), len(baseline))].mean()
        # For large extensions a normal approximation avoids a 203k x 500 bootstrap matrix.
        b = nonoverlap.mean() + rng.normal() * nonoverlap.std(ddof=1) / np.sqrt(len(nonoverlap))
        boot.append(b - a)
    return {"domain": domain, "A1_n": len(baseline), "extension_n": len(extension),
            "id_overlap_count": overlap, "A1_mean_Q0": float(baseline.mean()),
            "extension_mean_Q0": float(extension.mean()), "delta_extension_minus_A1": float(extension.mean() - baseline.mean()),
            "nonoverlap_n": len(nonoverlap), "nonoverlap_mean_Q0": float(nonoverlap.mean()),
            "delta_nonoverlap_minus_A1": float(nonoverlap.mean() - baseline.mean()),
            "delta_ci95_low_approx": float(np.quantile(boot, .025)),
            "delta_ci95_high_approx": float(np.quantile(boot, .975)),
            "wasserstein_W1": float(wasserstein_distance(baseline, extension)),
            "A1_median_Q0": float(np.median(baseline)),
            "extension_median_Q0": float(np.median(extension)),
            "inference_note": "Difference interval compares disjoint IDs, but both parts are from the same underlying source/file."}


def main() -> None:
    params = json.loads((OUT / "A1_frozen_scale.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((OUT / "A1_frozen_hierarchy.json").read_text(encoding="utf-8"))
    group_weights = {int(k): v for k, v in hierarchy["group_weights"].items()}
    with np.load(OUT / "A1_frozen_ecdf.npz") as archive:
        references = {field: [archive[f"{field}__{i}"].copy() for i in range(7)] for field in DIRECTIONS}
    baseline = pd.read_csv(OUT / "A1_sample_Q0.csv", usecols=["id", "domain", "Q0"])
    results = []
    histograms = {}

    for domain, path in source_files().items():
        a1_group = baseline[baseline.domain == domain]
        a1_ids = set(a1_group.id.astype(str))
        ids_seen: set[str] = set()
        duplicate_ids = 0
        overlap = 0
        all_scores: list[np.ndarray] = []
        rest_scores: list[np.ndarray] = []
        batch: list[dict[str, object]] = []
        batch_overlap: list[bool] = []
        target = OUT / f"A{'2' if domain == 'arxiv' else '3'}_sample_Q0.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "domain", *[f"H_{k}" for k in sorted(group_weights)], "Q0"])

            def flush() -> None:
                if not batch:
                    return
                chunk = pd.DataFrame(batch)
                standardized = score_frame(chunk, references, params["median_fallback_global"],
                                           params["median_fallback_by_domain"])
                scored = score_clusters(standardized, group_weights)
                if scored.Q0.isna().any() or not scored.Q0.between(0, 1).all():
                    raise ValueError(f"Invalid {domain} extension scores")
                all_scores.append(scored.Q0.to_numpy())
                rest_scores.append(scored.Q0.to_numpy()[~np.asarray(batch_overlap, dtype=bool)])
                for identifier, row in zip(chunk.id, scored.itertuples(index=False, name=None)):
                    writer.writerow([identifier, domain, *[format(v, ".10g") for v in row]])
                batch.clear()
                batch_overlap.clear()

            stream = lzma.open(path, "rt", encoding="utf-8") if path.suffix == ".xz" else path.open(encoding="utf-8")
            with stream:
                for line in stream:
                    record = json.loads(line)
                    identifier = str(record.get("id", ""))
                    if not identifier:
                        raise ValueError(f"Missing ID in {path}")
                    if identifier in ids_seen:
                        duplicate_ids += 1
                    ids_seen.add(identifier)
                    is_overlap = identifier in a1_ids
                    overlap += is_overlap
                    batch_overlap.append(is_overlap)
                    signals, _ = scalarize(record)
                    batch.append({"id": identifier, "domain": domain, **signals})
                    if len(batch) >= 4000:
                        flush()
                flush()
        values = np.concatenate(all_scores)
        rest = np.concatenate(rest_scores)
        baseline_values = a1_group.Q0.to_numpy()
        results.append({**compare(domain, baseline_values, values, rest, overlap), "duplicate_ids_in_extension": duplicate_ids,
                        "source_path": str(path)})
        histograms[domain] = (baseline_values, values)
        print(f"{domain}: {len(values):,} scored, {overlap:,} A1 ID matches, {duplicate_ids} duplicate IDs")

    pd.DataFrame(results).to_csv(OUT / "extension_representativeness.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, (domain, (a1, expanded)) in zip(axes, histograms.items()):
        sns.kdeplot(a1, label=f"A1 ({len(a1):,})", color="#c06d4b", ax=ax)
        sns.kdeplot(expanded, label=f"Extension ({len(expanded):,})", color="#216e77", ax=ax)
        ax.set(title=domain, xlabel="Frozen-model Q0", xlim=(0, 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_vs_A2_A3_Q0.png", dpi=200)
    plt.close(fig)
    print(pd.DataFrame(results)[["domain", "A1_n", "extension_n", "id_overlap_count",
                                 "delta_extension_minus_A1", "delta_nonoverlap_minus_A1", "wasserstein_W1"]].to_string(index=False))


if __name__ == "__main__":
    main()
