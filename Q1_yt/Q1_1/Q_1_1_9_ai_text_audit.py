"""Record a small, explicitly AI-assisted face-validity audit of A1 text."""

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "results"

# Observations are based on reading the first 2,500 characters of each text.
# These are qualitative case notes, not blind or independent human labels.
CASES = [
    ("TXT-036", "Coherent scientific introduction despite low Q0", "possible under-scoring of LaTeX/scientific text"),
    ("TXT-058", "Coherent physics introduction", "face-valid high score; factual correctness unverified"),
    ("TXT-002", "Copyright and front matter rather than substantive passage", "face-valid low score for standalone training value"),
    ("TXT-007", "Structured book contents and introduction", "high score reflects organization; not necessarily information density"),
    ("TXT-054", "Fragmented quoted mailing-list exchange", "face-valid low score for fragmentary excerpt"),
    ("TXT-018", "Coherent arts essay", "face-valid high score for readable prose"),
    ("TXT-011", "Promotional DJ profile", "possible ad/promotion contamination at low score"),
    ("TXT-009", "Coherent report on television representation", "face-valid high score for structured reporting"),
    ("TXT-003", "Valid PHP interface declaration", "possible under-scoring of concise usable code"),
    ("TXT-005", "Structured Vagrant setup instructions", "face-valid high score for actionable documentation"),
    ("TXT-015", "Specific JavaFX question with configuration details", "low score does not prove low technical usefulness"),
    ("TXT-016", "Concise iOS question and linked answer", "link-dependent answer cannot be checked from excerpt"),
    ("TXT-032", "Detailed network troubleshooting question without confirmed diagnosis", "high score does not establish factual correctness or answer quality"),
    ("TXT-001", "Russian asteroid table with structured factual rows", "low prose score may penalize useful multilingual tables"),
    ("TXT-008", "Short, coherent Swedish taxonomy entry", "short multilingual stub may be useful despite sparse prose"),
    ("TXT-004", "PK disambiguation list", "high score need not imply rich explanatory content"),
]


def main() -> None:
    blind = pd.read_csv(OUT / "A1_manual_review_BLIND.csv", keep_default_na=False)
    key = pd.read_csv(OUT / "A1_manual_review_SEALED_KEY.csv")
    if any(blind[col].astype(str).str.strip().ne("").any() for col in
           ["readability_1to5", "information_1to5", "absence_of_noise_1to5", "completeness_1to5"]):
        raise ValueError("The human blind-review form must remain unscored")
    merged = blind.merge(key, on=["review_id", "domain"], validate="one_to_one")
    entries = []
    for review_id, observation, caveat in CASES:
        row = merged.loc[merged.review_id == review_id]
        if len(row) != 1 or not str(row.iloc[0].content).strip():
            raise ValueError(f"Missing reviewed text: {review_id}")
        item = row.iloc[0]
        entries.append({"review_id": review_id, "domain": item.domain,
                        "score_stratum": item.score_stratum, "Q0": item.Q0,
                        "text_truncated": item.text_truncated,
                        "observation": observation, "interpretation_or_limitation": caveat,
                        "evidence_type": "AI-assisted qualitative reading; not human validation"})
    cases = pd.DataFrame(entries)
    if cases.domain.nunique() != 7:
        raise ValueError("Qualitative case review must cover all seven domains")
    cases.to_csv(OUT / "AI_assisted_text_audit_cases.csv", index=False, encoding="utf-8-sig")
    lines = ["# AI-assisted qualitative text audit", "",
             "This is a non-blind AI-assisted reading of 16 selected texts from the 210-row A1 review package,",
             "not an independent human validation or a random population estimate. Selection uses the sealed",
             "score key; observations use only the first 2,500 characters. The 210 human score cells stay blank.",
             "No human-score correlation, inter-rater agreement, or factual-accuracy claim is made.", "",
             "| Review ID | Domain | Stratum | Q0 | Observed excerpt type | Implication |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for item in cases.itertuples(index=False):
        lines.append(f"| {item.review_id} | {item.domain} | {item.score_stratum} | "
                     f"{item.Q0:.3f} | {item.observation} | {item.interpretation_or_limitation} |")
    lines += ["", "Interpretation: several high/low contrasts are plausible, but code, mathematical notation,",
              "multilingual tables, short factual stubs and question-only pages expose construct-validity risks.",
              "Model scores measure agreement with the chosen automated signals, not truth, correctness, or",
              "downstream language-model utility. Do not write 'human review confirms Q0' in the paper.", "",
              "Disclosure: AI assisted with selecting, reading and describing these qualitative cases;",
              "there was no independent human annotation of this sample."]
    (OUT / "AI_assisted_text_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AI-assisted audit: {len(cases)} texts across {cases.domain.nunique()} domains; no human labels")


if __name__ == "__main__":
    main()
