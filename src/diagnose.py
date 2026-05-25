"""
Diagnose why 5 specific factoid queries fail.
For each query: show top-10 retrieved chunks + check if chunks containing the key term were retrieved at all.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import Retriever

CHUNKS_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")

QUERIES = [
    {
        "q": "PD-31 数据集有多少被试，分别是什么类别？",
        "keyword": "PD-31",
        "expected_answer_contains": ["31", "16", "15", "PD"],
    },
    {
        "q": "ADHD-200 数据集使用多少 ROI，基于什么脑图谱？",
        "keyword": "ADHD-200",
        "expected_answer_contains": ["ADHD-200", "ROI", "atlas"],
    },
    {
        "q": "CBraMod 在 BCIC2020-3 数据集上的 Balanced Accuracy 是多少？",
        "keyword": "BCIC2020-3",
        "expected_answer_contains": ["BCIC2020-3", "Balanced", "CBraMod"],
    },
    {
        "q": "Brain-OF 的 ADHD-200 benchmark 里 FFCL 的分数是多少？",
        "keyword": "FFCL",
        "expected_answer_contains": ["FFCL", "ADHD-200"],
    },
    {
        "q": "BrainWave 论文中 ADHD-Adult 数据集有多少样本？",
        "keyword": "ADHD-Adult",
        "expected_answer_contains": ["ADHD-Adult"],
    },
]


def find_chunks_with_keyword(keyword: str) -> list[dict]:
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    return [c for c in chunks if keyword in c["text"]]


def main():
    print("Loading retriever...", flush=True)
    r = Retriever()
    print()

    K = 10
    for item in QUERIES:
        q = item["q"]
        kw = item["keyword"]
        print("=" * 90)
        print(f"Q: {q}")
        print(f"   keyword: '{kw}'")

        # 哪些 chunks 真的包含这个关键词
        gold_chunks = find_chunks_with_keyword(kw)
        gold_ids = {c["chunk_id"] for c in gold_chunks}
        print(f"   '{kw}' 出现在 {len(gold_chunks)} 个 chunks（chunk_ids: {sorted(gold_ids)[:8]}{'...' if len(gold_ids) > 8 else ''}）")

        # 跑 top-10 检索
        hits = r.search(q, k=K)
        hit_ids = [h["chunk_id"] for h in hits]
        gold_in_topk = [i for i, h in enumerate(hits, 1) if h["chunk_id"] in gold_ids]

        if gold_in_topk:
            print(f"   ✓ top-{K} 里包含含关键词的 chunk：rank {gold_in_topk}")
        else:
            print(f"   ✗ top-{K} 里**没有任何**包含关键词 '{kw}' 的 chunk")

        print(f"\n   Top-{K} chunks:")
        for i, h in enumerate(hits, 1):
            mark = "★" if h["chunk_id"] in gold_ids else " "
            snippet = h["text"].replace("\n", " ").strip()[:130]
            print(f"   {mark} [{i}] score={h['score']:.3f} id={h['chunk_id']:>4}  {h['source'][:50]}")
            print(f"        {snippet}")
        print()


if __name__ == "__main__":
    main()
