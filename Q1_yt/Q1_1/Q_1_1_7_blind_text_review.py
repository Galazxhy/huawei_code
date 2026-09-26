"""Prepare a blinded 210-text manual review without fabricating human labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Q_1_1_1_audit_A1 import locate_a1, open_jsonl


OUT = Path(__file__).resolve().parent / "results"
SEED = 20260923


def main() -> None:
    scores = pd.read_csv(OUT / "A1_sample_Q0.csv", usecols=["id", "domain", "Q0"])
    audited = pd.read_csv(OUT / "A1_scalarized_signals.csv", usecols=["id", "content_length"])
    available = scores.merge(audited, on="id", validate="one_to_one")
    rng = np.random.default_rng(SEED)
    targets = []
    for domain, group in available.groupby("domain", sort=True):
        # Enforce a readable manual-review budget while retaining the score-stratum ranks.
        eligible = group[group.content_length.between(100, 20000)].sort_values("Q0")
        if len(eligible) < 100:
            eligible = group.sort_values("Q0")
        strata = {
            "low": eligible.head(max(10, int(len(eligible) * .10))),
            "middle": eligible.iloc[max(0, len(eligible) // 2 - max(10, int(len(eligible) * .05))):
                                     min(len(eligible), len(eligible) // 2 + max(10, int(len(eligible) * .05)))],
            "high": eligible.tail(max(10, int(len(eligible) * .10))),
        }
        for stratum, candidates in strata.items():
            selected = candidates.sample(n=10, random_state=int(rng.integers(0, 2**31 - 1)))
            for row in selected.itertuples(index=False):
                targets.append({"id": row.id, "domain": domain, "stratum": stratum,
                                "Q0": float(row.Q0), "content_length": int(row.content_length)})
    if len(targets) != 210 or len({row["id"] for row in targets}) != 210:
        raise ValueError("Manual-review selection is not 210 distinct texts")

    selected_ids = {row["id"] for row in targets}
    content: dict[str, str] = {}
    with open_jsonl(locate_a1()) as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("id") in selected_ids:
                content[record["id"]] = str(record.get("content", ""))
    if set(content) != selected_ids:
        raise ValueError(f"Missing {len(selected_ids - set(content))} review texts")

    rng.shuffle(targets)
    blind_rows = []
    key_rows = []
    for index, row in enumerate(targets, start=1):
        review_id = f"TXT-{index:03d}"
        full_text = content[row["id"]]
        blind_rows.append({"review_id": review_id, "domain": row["domain"],
                           "content": full_text[:2500], "text_truncated": len(full_text) > 2500,
                           "readability_1to5": "", "information_1to5": "",
                           "absence_of_noise_1to5": "", "completeness_1to5": "",
                           "reviewer_id": "", "review_notes": ""})
        key_rows.append({"review_id": review_id, "source_id": row["id"], "domain": row["domain"],
                         "score_stratum": row["stratum"], "Q0": row["Q0"],
                         "original_chars": len(full_text)})
    pd.DataFrame(blind_rows).to_csv(OUT / "A1_manual_review_BLIND.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key_rows).to_csv(OUT / "A1_manual_review_SEALED_KEY.csv", index=False, encoding="utf-8-sig")
    rubric = """# 原文盲审评分说明\n\n该文件是人工审阅协议，不含人工结论。两名评阅者分别复制盲审表填写，直到两人完成前不打开 SEALED_KEY。\n\n每项用 1–5 整数计分（1=明显不符合，3=部分符合，5=明确符合）：\n\n1. 可读性：语言/代码是否清晰、有可理解的结构。\n2. 信息有效性：是否包含可辨认且有用的事实、解释、推导或代码。\n3. 无噪声与广告：乱码、重复、广告等是否较少。\n4. 完整性：片段是否足以表达一个相对完整的内容单元。\n\n如果原文超过 2500 字符，盲审表仅包含前 2500 字符并标注截断。遇到必须看全文的样本，以密封答案键的原始 id 回查 A1，但在评分提交前不可读取 Q0。专业文本、代码与自然语言的合理写法不同，不可简单以英语散文为唯一标准。\n\n所有 210 条按域分别选高/中/低三个 Q0 层，每层各 10 条；分层只用于抽样和事后检验，不代表人工类别。人工审核完成后用两人的四项均分与模型 Q0 计算 Spearman，并报告两人的一致性（如 Kendall W 或 ICC）及盲审限制。\n"""
    (OUT / "A1_manual_review_rubric.md").write_text(rubric, encoding="utf-8")
    print(f"Prepared {len(blind_rows)} blinded review rows; human scores remain blank.")


if __name__ == "__main__":
    main()
