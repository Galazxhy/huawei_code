"""Complete the 210-row review with an explicitly AI-assisted reading.

This is NOT a human blind review and must be disclosed as such in the paper.
The script records one AI-assisted reviewer, fills the four rubric scores,
and reports face-validity correlation with the model score Q0. Because the
210 texts were sampled by Q0 stratum, the correlation is a conditional
face-validity check, not a random-population validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent / "results"

# (readability, information, absence_of_noise, completeness), all 1-5.
AI_SCORES = {
    "TXT-001": (3, 3, 4, 2), "TXT-002": (4, 1, 5, 2), "TXT-003": (4, 4, 5, 4),
    "TXT-004": (4, 4, 5, 4), "TXT-005": (4, 4, 5, 4), "TXT-006": (3, 2, 4, 2),
    "TXT-007": (4, 2, 5, 2), "TXT-008": (4, 3, 5, 3), "TXT-009": (4, 5, 5, 4),
    "TXT-010": (4, 2, 5, 2), "TXT-011": (3, 2, 3, 3), "TXT-012": (4, 2, 5, 3),
    "TXT-013": (4, 4, 5, 4), "TXT-014": (4, 3, 5, 4), "TXT-015": (3, 5, 4, 4),
    "TXT-016": (4, 4, 5, 4), "TXT-017": (4, 5, 5, 4), "TXT-018": (5, 4, 5, 4),
    "TXT-019": (3, 2, 4, 3), "TXT-020": (4, 2, 5, 3), "TXT-021": (4, 3, 5, 3),
    "TXT-022": (4, 4, 5, 4), "TXT-023": (4, 2, 5, 3), "TXT-024": (4, 4, 5, 4),
    "TXT-025": (4, 4, 4, 4), "TXT-026": (4, 4, 5, 4), "TXT-027": (3, 1, 4, 1),
    "TXT-028": (4, 4, 5, 4), "TXT-029": (5, 3, 5, 3), "TXT-030": (4, 4, 5, 4),
    "TXT-031": (3, 2, 3, 3), "TXT-032": (4, 4, 5, 3), "TXT-033": (4, 4, 5, 4),
    "TXT-034": (4, 3, 5, 3), "TXT-035": (4, 4, 5, 4), "TXT-036": (4, 5, 5, 4),
    "TXT-037": (4, 5, 5, 4), "TXT-038": (4, 3, 5, 3), "TXT-039": (3, 3, 3, 3),
    "TXT-040": (3, 2, 3, 3), "TXT-041": (4, 5, 5, 4), "TXT-042": (4, 2, 5, 3),
    "TXT-043": (4, 5, 5, 4), "TXT-044": (3, 3, 4, 3), "TXT-045": (4, 5, 5, 4),
    "TXT-046": (4, 4, 5, 4), "TXT-047": (4, 4, 5, 4), "TXT-048": (3, 4, 5, 3),
    "TXT-049": (4, 3, 4, 4), "TXT-050": (4, 2, 5, 2), "TXT-051": (3, 4, 4, 4),
    "TXT-052": (4, 5, 5, 4), "TXT-053": (3, 5, 5, 4), "TXT-054": (1, 1, 2, 1),
    "TXT-055": (4, 3, 5, 3), "TXT-056": (4, 3, 5, 3), "TXT-057": (4, 4, 4, 4),
    "TXT-058": (4, 5, 5, 4), "TXT-059": (5, 3, 5, 3), "TXT-060": (4, 5, 5, 4),
    "TXT-061": (4, 4, 5, 4), "TXT-062": (3, 4, 4, 4), "TXT-063": (4, 2, 5, 2),
    "TXT-064": (4, 3, 5, 3), "TXT-065": (2, 1, 1, 2), "TXT-066": (3, 2, 4, 3),
    "TXT-067": (4, 4, 4, 4), "TXT-068": (4, 4, 5, 4), "TXT-069": (2, 3, 3, 2),
    "TXT-070": (3, 3, 5, 3), "TXT-071": (3, 2, 5, 3), "TXT-072": (4, 3, 5, 3),
    "TXT-073": (3, 3, 4, 4), "TXT-074": (2, 2, 4, 2), "TXT-075": (4, 4, 5, 4),
    "TXT-076": (3, 2, 4, 3), "TXT-077": (4, 2, 5, 3), "TXT-078": (3, 4, 4, 4),
    "TXT-079": (4, 3, 5, 3), "TXT-080": (4, 5, 5, 4), "TXT-081": (5, 4, 5, 4),
    "TXT-082": (2, 4, 4, 3), "TXT-083": (3, 2, 5, 3), "TXT-084": (4, 4, 5, 3),
    "TXT-085": (3, 2, 4, 2), "TXT-086": (3, 3, 4, 2), "TXT-087": (3, 4, 4, 4),
    "TXT-088": (4, 3, 5, 3), "TXT-089": (4, 5, 5, 4), "TXT-090": (2, 2, 2, 2),
    "TXT-091": (4, 5, 5, 4), "TXT-092": (4, 4, 5, 4), "TXT-093": (4, 5, 5, 3),
    "TXT-094": (4, 4, 5, 4), "TXT-095": (4, 3, 5, 3), "TXT-096": (4, 5, 5, 4),
    "TXT-097": (4, 4, 5, 4), "TXT-098": (4, 5, 5, 4), "TXT-099": (3, 2, 4, 3),
    "TXT-100": (3, 4, 5, 4), "TXT-101": (4, 5, 5, 4), "TXT-102": (4, 3, 5, 3),
    "TXT-103": (3, 3, 4, 4), "TXT-104": (3, 4, 5, 4), "TXT-105": (4, 4, 5, 4),
    "TXT-106": (4, 3, 5, 3), "TXT-107": (4, 3, 5, 4), "TXT-108": (4, 5, 5, 4),
    "TXT-109": (4, 5, 5, 4), "TXT-110": (4, 4, 5, 4), "TXT-111": (4, 2, 5, 2),
    "TXT-112": (4, 2, 5, 2), "TXT-113": (3, 2, 5, 2), "TXT-114": (4, 3, 5, 3),
    "TXT-115": (3, 5, 5, 4), "TXT-116": (4, 5, 5, 4), "TXT-117": (4, 3, 5, 4),
    "TXT-118": (4, 4, 5, 4), "TXT-119": (4, 4, 5, 4), "TXT-120": (4, 4, 5, 4),
    "TXT-121": (4, 3, 5, 4), "TXT-122": (4, 3, 5, 3), "TXT-123": (2, 2, 4, 2),
    "TXT-124": (4, 4, 5, 4), "TXT-125": (4, 2, 5, 3), "TXT-126": (4, 5, 5, 4),
    "TXT-127": (4, 4, 5, 4), "TXT-128": (4, 5, 5, 4), "TXT-129": (4, 3, 5, 4),
    "TXT-130": (4, 2, 5, 2), "TXT-131": (3, 2, 4, 3), "TXT-132": (3, 2, 2, 3),
    "TXT-133": (4, 4, 4, 4), "TXT-134": (4, 3, 5, 3), "TXT-135": (4, 4, 5, 4),
    "TXT-136": (4, 4, 5, 4), "TXT-137": (3, 1, 2, 3), "TXT-138": (1, 1, 1, 1),
    "TXT-139": (4, 4, 4, 4), "TXT-140": (3, 1, 5, 2), "TXT-141": (4, 5, 5, 4),
    "TXT-142": (3, 3, 4, 3), "TXT-143": (4, 2, 5, 2), "TXT-144": (1, 1, 1, 1),
    "TXT-145": (3, 3, 5, 2), "TXT-146": (4, 5, 5, 4), "TXT-147": (3, 2, 5, 3),
    "TXT-148": (4, 4, 5, 4), "TXT-149": (2, 1, 1, 2), "TXT-150": (4, 5, 5, 4),
    "TXT-151": (3, 2, 5, 2), "TXT-152": (3, 4, 4, 4), "TXT-153": (4, 4, 5, 4),
    "TXT-154": (4, 5, 5, 4), "TXT-155": (4, 3, 5, 3), "TXT-156": (4, 4, 5, 4),
    "TXT-157": (4, 4, 5, 4), "TXT-158": (3, 4, 5, 4), "TXT-159": (3, 3, 4, 3),
    "TXT-160": (4, 5, 5, 4), "TXT-161": (4, 4, 4, 4), "TXT-162": (4, 5, 5, 4),
    "TXT-163": (3, 3, 5, 3), "TXT-164": (3, 4, 4, 4), "TXT-165": (3, 5, 5, 4),
    "TXT-166": (3, 3, 3, 3), "TXT-167": (3, 2, 4, 2), "TXT-168": (4, 5, 5, 4),
    "TXT-169": (3, 1, 5, 2), "TXT-170": (4, 4, 5, 4), "TXT-171": (4, 3, 4, 4),
    "TXT-172": (3, 2, 5, 3), "TXT-173": (4, 4, 5, 4), "TXT-174": (4, 3, 5, 4),
    "TXT-175": (3, 2, 5, 3), "TXT-176": (3, 4, 4, 4), "TXT-177": (4, 3, 5, 3),
    "TXT-178": (4, 4, 5, 4), "TXT-179": (3, 2, 5, 2), "TXT-180": (4, 4, 5, 4),
    "TXT-181": (4, 5, 5, 4), "TXT-182": (4, 2, 5, 3), "TXT-183": (4, 5, 5, 4),
    "TXT-184": (3, 2, 5, 2), "TXT-185": (3, 5, 5, 4), "TXT-186": (4, 4, 5, 4),
    "TXT-187": (4, 5, 5, 4), "TXT-188": (4, 5, 5, 4), "TXT-189": (4, 4, 5, 4),
    "TXT-190": (2, 1, 3, 2), "TXT-191": (4, 5, 5, 4), "TXT-192": (3, 3, 5, 3),
    "TXT-193": (4, 4, 5, 4), "TXT-194": (3, 2, 5, 2), "TXT-195": (4, 5, 5, 4),
    "TXT-196": (3, 2, 4, 3), "TXT-197": (3, 2, 5, 2), "TXT-198": (4, 4, 5, 4),
    "TXT-199": (4, 5, 5, 4), "TXT-200": (2, 1, 4, 2), "TXT-201": (4, 5, 5, 4),
    "TXT-202": (4, 2, 4, 3), "TXT-203": (4, 4, 5, 4), "TXT-204": (3, 4, 4, 4),
    "TXT-205": (4, 3, 5, 3), "TXT-206": (3, 2, 5, 2), "TXT-207": (3, 2, 5, 2),
    "TXT-208": (3, 3, 4, 4), "TXT-209": (3, 4, 5, 4), "TXT-210": (4, 5, 5, 4),
}


def main() -> None:
    key = pd.read_csv(OUT / "A1_manual_review_SEALED_KEY.csv")
    blind = pd.read_csv(OUT / "A1_manual_review_BLIND.csv", keep_default_na=False)
    if set(AI_SCORES) != set(key.review_id):
        missing = set(key.review_id) - set(AI_SCORES)
        raise ValueError(f"AI review missing cases: {missing}")

    rows = []
    for review_id, scores in AI_SCORES.items():
        rows.append({
            "review_id": review_id,
            "domain": key.loc[key.review_id == review_id, "domain"].iloc[0],
            "readability_1to5": scores[0],
            "information_1to5": scores[1],
            "absence_of_noise_1to5": scores[2],
            "completeness_1to5": scores[3],
            "AI_mean": float(np.mean(scores)),
            "reviewer_id": "AI_assisted_reviewer",
            "review_notes": "AI-assisted reading of first 2500 chars; explicitly not a human blind score",
        })
    reviewed = pd.DataFrame(rows)
    reviewed.to_csv(OUT / "A1_AI_review_scores.csv", index=False, encoding="utf-8-sig")

    merged = reviewed.merge(key, on=["review_id", "domain"], validate="one_to_one")
    overall = float(spearmanr(merged.AI_mean, merged.Q0).statistic)
    by_domain = []
    for domain, group in merged.groupby("domain", sort=True):
        by_domain.append({
            "domain": domain,
            "n": len(group),
            "spearman_AI_mean_vs_Q0": float(spearmanr(group.AI_mean, group.Q0).statistic),
            "mean_AI_mean": float(group.AI_mean.mean()),
            "mean_Q0": float(group.Q0.mean()),
        })
    domain_table = pd.DataFrame(by_domain)
    domain_table.to_csv(OUT / "A1_AI_review_by_domain.csv", index=False, encoding="utf-8-sig")

    summary = {
        "review_type": "AI-assisted, not human blind review",
        "n": len(merged),
        "spearman_AI_mean_vs_Q0_overall": overall,
        "sampling_note": "210 cases are Q0-stratified, so this correlation is a conditional face-validity check.",
    }
    (OUT / "A1_AI_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=merged, x="Q0", y="AI_mean", hue="domain", ax=ax, alpha=0.7)
    ax.set(xlabel="Model Q0", ylabel="AI-assisted rubric mean (1-5 -> normalized)",
           title=f"AI-assisted face validity vs Q0 (Spearman={overall:.3f})")
    ax.set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A1_AI_review_face_validity.png", dpi=200)
    plt.close(fig)

    print(f"AI-assisted review complete: {len(rows)} cases")
    print(f"Overall Spearman(AI_mean, Q0) = {overall:.3f}")
    print(domain_table.to_string(index=False))


if __name__ == "__main__":
    main()
