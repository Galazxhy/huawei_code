"""Build a complete 22-indicator audit table from the standardized signals.

The table is one row per original quality field, using the standardized
domain-balanced suitability score s in [0,1] produced by Q_1_1_3_freeze_scale.py.
It complements Q_1_1_1_audit_A1.py, which records raw field types and list
structure but deliberately does not invent a distribution for list-valued
fields before their documented scalarization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "results"

# Original-field metadata. Cluster labels are loaded from the frozen clustering
# output; direction labels come from the frozen scale. Invalid/imputed counts are
# only defined for transformed model logits.
FIELD_ORDER = [
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

TYPE_AND_TRANSFORM = {
    "fineweb_edu": ("list(1)", "取唯一教育评分"),
    "fluency_en": ("list(2)", "softmax 取 P(fluent)"),
    "modernbert_cleanliness": ("list(6)", "6 级 logits 期望评级/5"),
    "modernbert_readability": ("list(6)", "6 级 logits 期望评级/5"),
    "modernbert_reasoning": ("list(6)", "6 级 logits 期望评级/5"),
    "modernbert_professionalism": ("list(6)", "6 级 logits 期望评级/5"),
    "dsir_books": ("scalar", "直接保留"),
    "dsir_wiki": ("scalar", "直接保留"),
    "dsir_math": ("scalar", "直接保留"),
    "qurater": ("list(4)", "四分量各自百分位后取均值"),
    "ad_en": ("list(2)", "softmax 取 P(no_ad)"),
    "rps_doc_word_count": ("scalar", "直接保留"),
    "rps_doc_num_sentences": ("scalar", "直接保留"),
    "rps_doc_unigram_entropy": ("scalar", "直接保留"),
    "rps_doc_frac_unique_words": ("scalar", "直接保留"),
    "rps_doc_frac_no_alph_words": ("scalar", "直接保留"),
    "rps_doc_frac_chars_top_2gram": ("scalar", "直接保留"),
    "rps_doc_frac_chars_top_3gram": ("scalar", "直接保留"),
    "rps_lines_uppercase_letter_fraction": ("scalar", "直接保留"),
    "rps_lines_ending_with_terminal_punctution_mark": ("scalar", "直接保留"),
    "rps_lines_numerical_chars_fraction": ("scalar", "直接保留"),
    "rps_doc_mean_word_length": ("scalar", "直接保留"),
}

DIRECTION_LABELS = {
    "positive": "正向",
    "negative": "负向",
    "middle": "中心适宜",
}

RISK = {
    "fineweb_edu": "教育性偏好不等于所有领域质量",
    "fluency_en": "偏向英语散文，对代码/多语言不友好",
    "modernbert_cleanliness": "自动模型偏差",
    "modernbert_readability": "可读性不等于专业性",
    "modernbert_reasoning": "非推理文本可能被误伤",
    "modernbert_professionalism": "风格依赖，含 5 个无效值",
    "dsir_books": "领域倾向不等于质量",
    "dsir_wiki": "领域倾向不等于质量",
    "dsir_math": "领域倾向不等于质量",
    "qurater": "expertise 中心适宜，其余正向；均值合成",
    "ad_en": "广告分类器有领域偏差",
    "rps_doc_word_count": "篇幅不是直接质量",
    "rps_doc_num_sentences": "代码/列表的句子数不适用",
    "rps_doc_unigram_entropy": "过低重复、过高可能噪声",
    "rps_doc_frac_unique_words": "受语言和篇幅影响",
    "rps_doc_frac_no_alph_words": "代码、公式、非拉丁文误伤",
    "rps_doc_frac_chars_top_2gram": "重复字符代理",
    "rps_doc_frac_chars_top_3gram": "重复字符代理",
    "rps_lines_uppercase_letter_fraction": "大写格式可能有正当用途",
    "rps_lines_ending_with_terminal_punctution_mark": "散文标点代理，代码不适用",
    "rps_lines_numerical_chars_fraction": "数字密度依学科而变",
    "rps_doc_mean_word_length": "长术语及语言偏差",
}


def describe(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(finite.size),
        "min": float(finite.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "mean": float(finite.mean()),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(finite.max()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    }


def main() -> None:
    frame = pd.read_csv(OUT / "A1_directional_signals.csv")
    clusters = pd.read_csv(OUT / "A1_indicator_clusters.csv").set_index("indicator")["cluster"]
    params = json.loads((OUT / "A1_frozen_scale.json").read_text(encoding="utf-8"))
    scalarization = json.loads((OUT / "A1_scalarization_summary.json").read_text(encoding="utf-8"))
    invalid_counts = scalarization.get("missing_or_invalid_scalar_counts", {})

    rows = []
    for field in FIELD_ORDER:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        stats = describe(values)
        domain_medians = (
            frame.groupby("domain")[field].median()
        )
        direction = "mixed" if field == "qurater" else params["directions"].get(field, "unknown")
        direction_text = (
            "混合（expertise 中心适宜，其余正向）"
            if field == "qurater"
            else DIRECTION_LABELS.get(direction, direction)
        )
        raw_type, transform = TYPE_AND_TRANSFORM[field]
        rows.append({
            "原始指标": field,
            "原始类型": raw_type,
            "标量化/变换": transform,
            "方向": direction_text,
            "簇": int(clusters.loc[field]) if field in clusters.index else None,
            "无效或填补记录": int(invalid_counts.get(field, 0)),
            "样本量": stats["n"],
            "极小值s": round(stats["min"], 4),
            "P05_s": round(stats["p05"], 4),
            "中位数s": round(stats["median"], 4),
            "均值s": round(stats["mean"], 4),
            "P95_s": round(stats["p95"], 4),
            "极大值s": round(stats["max"], 4),
            "标准差s": round(stats["std"], 4),
            "七域中位数_min": round(float(domain_medians.min()), 4),
            "七域中位数_max": round(float(domain_medians.max()), 4),
            "风险提示": RISK[field],
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "A1_standardized_indicator_audit.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "# A1 22 个质量指标标准化审计总表",
        "",
        "该表按**原始字段**计数，`s` 为冻结模型中的域均衡适宜度分，均落在 [0,1] 且方向已统一为“越高越好”。",
        "`qurater` 内部四个语义分量先各自做百分位变换再平均，表中给的是合并后的单一原始字段信号。",
        "“无效或填补记录”只对模型 logits 型字段有意义；其余字段原始值完整。",
        "",
        "| 原始指标 | 原始类型/变换 | 方向 | 簇 | 无效填补 | 中位数s | 均值s | 标准差s | P05_s | P95_s | 七域中位数区间 | 风险提示 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['原始指标']} | {row['原始类型']}/{row['标量化/变换']} | {row['方向']} | "
            f"{row['簇']} | {row['无效或填补记录']} | {row['中位数s']} | {row['均值s']} | "
            f"{row['标准差s']} | {row['P05_s']} | {row['P95_s']} | "
            f"[{row['七域中位数_min']}, {row['七域中位数_max']}] | {row['风险提示']} |"
        )
    (OUT / "A1_22指标审计总表.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Standardized 22-indicator audit table written for {len(rows)} original fields.")


if __name__ == "__main__":
    main()
