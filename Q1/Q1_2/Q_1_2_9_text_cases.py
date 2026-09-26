"""Assemble explainable high-conflict text cases (AI-assisted reading)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
_A1_DIR = ROOT.parents[1] / "data" / "real_attachments" / "A_data_value"
# The attachment ships as a flat JSONL; older copies nested it in a directory.
A1_FILE = _A1_DIR / "slimpajama_quality_signal_sample.jsonl"
if A1_FILE.is_dir():
    A1_FILE = A1_FILE / "slimpajama_quality_signal_sample.jsonl"
elif not A1_FILE.exists():
    _nested = _A1_DIR / "slimpajama_quality_signal_sample.jsonl" / "slimpajama_quality_signal_sample.jsonl"
    if _nested.exists():
        A1_FILE = _nested

CASE_EXPLANATIONS = {
    "BkiUdX3xK6wB9k0iLt8X": "LaTeX 宏/格式代码而非自然语言正文，句子数、词长、推理、专业性等散文质量指标与领域正常关系不一致。",
    "BkiUeBI4eIOjSKyN-CZl": "致谢/基金号与机构缩写密集，正式散文结构弱，触发句子数、大写、推理、专业性等结构指标冲突。",
    "BkiUdSLxK7IAD7XMJWdu": "古籍影印本版次/出版信息与早期排版，标点、唯一词、非字母、洁净度等指标偏离现代散文正常关系。",
    "BkiUasXxK1ThhCdy6D8f": "童书目录页，唯一词、3-gram、句子数、词数等结构指标因列表式目录而异常。",
    "BkiUdcc4uBhjDe9z1HI3": "大量“no record”模板句重复，熵、词数、唯一词、可读性、洁净度共同异常，属模板化网页。",
    "BkiUctk5qoTAhudwxeH1": "求职信模板/SEO 关键词堆砌，熵、词数、唯一词、非字母比例异常。",
    "BkiUdek4eIZjtO0fW2iK": "体育赛程/技术统计表，大量 0 与重复字段，熵、词数、唯一词、dsir_math、句子数异常。",
    "BkiUdNg5qhLA33KHMDZC": "曲棍球球员数据表，表格化数字使篇幅与词汇多样性指标偏离领域正常值。",
    "BkiUdhk4eIXhu5p8htwA": "TypeScript 导出常量，标识符与格式使大写、词数、熵、句子数、3-gram 冲突。",
    "BkiUdz84dbghePBCnbDg": "OpenGL 位域常量命名空间，代码标识符导致大写、词数、熵、句子数、3-gram 冲突。",
    "BkiUdpQ4dbghfTKFOa9k": "Magento 导入字段清单/错误日志，长列名与重复字段造成熵、词数、唯一词、大写、推理冲突。",
    "BkiUdas4eIXhzGirDH8U": "Tetris 二维数组表示问题，代码/矩阵表示导致熵、词数、唯一词结构冲突。",
    "BkiUeILxK1yAgWay5uDC": "菊石属名列表，长列表与拉丁学名导致熵、词数、唯一词、平均词长、句末标点异常。",
    "BkiUezHxK3YB9i3RMyMS": "小行星编号表格，表格化数据导致熵、词数、大写、平均词长、dsir_books 异常。",
}


def main() -> None:
    scores = pd.read_csv(OUT / "A1_conflict_scores.csv")
    signals = pd.read_csv(OUT / "A1_directional_signals.csv")
    signal_cols = list(signals.columns[2:])
    adjacency = pd.read_csv(OUT / "A1_quality_dependency_adjacency.csv", index_col=0)
    with np.load(OUT / "A1_anomaly.npz", allow_pickle=True) as archive:
        r = archive["r"]
        flag = archive["flag"]
        names = [str(x) for x in archive["names"]]

    selected = []
    for domain, group in scores.groupby("domain", sort=True):
        selected.extend(group.nlargest(2, "C_conflict").index.tolist())

    with open(A1_FILE, encoding="utf-8") as stream:
        content = {}
        for line in stream:
            record = json.loads(line)
            if record.get("id") in CASE_EXPLANATIONS:
                content[record.get("id")] = str(record.get("content", ""))
            if len(content) == len(CASE_EXPLANATIONS):
                break

    rows = []
    for idx in selected:
        row = scores.loc[idx]
        sample_id = str(row.id)
        rj = r[idx]
        fl = flag[idx]
        flagged = [(names[j], float(rj[j])) for j in range(len(names)) if fl[j]]
        flagged = sorted(flagged, key=lambda x: -x[1])
        top_conflict = "; ".join(f"{name}({val:.1f})" for name, val in flagged[:4])
        support_parts = []
        for name, _ in flagged[:3]:
            neighbors = [n for n in names if n != name and adjacency.loc[name, n] == 1]
            if neighbors:
                neighbor_values = {n: float(signals.loc[idx, n]) for n in neighbors}
                best = max(neighbor_values, key=neighbor_values.get)
                support_parts.append(f"{name}<-{best}({neighbor_values[best]:.2f})")
        rows.append({
            "sample_id": sample_id,
            "domain": row.domain,
            "C_conflict": round(float(row.C_conflict), 3),
            "n_conflict": int(row.n_conflict),
            "top_conflict_indicators": top_conflict,
            "top_support_indicators": "; ".join(support_parts[:4]),
            "content_excerpt": content.get(sample_id, "")[:300].replace("\n", " "),
            "AI_assisted_explanation": CASE_EXPLANATIONS.get(sample_id, ""),
            "evidence_type": "AI-assisted reading of high-conflict text; not human validation",
        })
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "conflict_text_cases.csv", index=False, encoding="utf-8-sig")

    md = ["# 典型高冲突原文案例（AI 辅助解释）", "",
          "| 样本 | 域 | C | 冲突指标 | 支持指标 | 原文特征 | 解释 |",
          "| --- | --- | ---: | --- | --- | --- | --- |"]
    for item in cases.itertuples(index=False):
        md.append(f"| {item.sample_id} | {item.domain} | {item.C_conflict} | "
                  f"{item.top_conflict_indicators} | {item.top_support_indicators} | "
                  f"{item.content_excerpt[:70]}... | {item.AI_assisted_explanation} |")
    md += ["", "说明：本表为 AI 辅助阅读，用于解释冲突成因，不代表真人盲审。"]
    (OUT / "Q1_2_冲突案例表.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"High-conflict text cases written: {len(cases)}")


if __name__ == "__main__":
    main()
