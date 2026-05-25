"""Re-run the 5 user queries through full RAG with hybrid + k=10."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).parent))
from qa import RAGSystem

QUERIES = [
    "PD-31 数据集有多少被试，分别是什么类别？",
    "ADHD-200 数据集使用多少 ROI，基于什么脑图谱？",
    "CBraMod 在 BCIC2020-3 数据集上的 Balanced Accuracy 是多少？",
    "Brain-OF 的 ADHD-200 benchmark 里 FFCL 的分数是多少？",
    "BrainWave 论文中 ADHD-Adult 数据集有多少样本？",
]


def main():
    print("Loading RAG system...")
    rag = RAGSystem(k=10, mode="rerank")
    print()

    for q in QUERIES:
        result = rag.answer(q)
        print("=" * 90)
        print(f"Q: {q}")
        print("-" * 90)
        print(f"A: {result['answer']}")
        print(f"\n  Top-10 chunks (hybrid):")
        for i, h in enumerate(result["hits"], 1):
            snippet = h["text"].replace("\n", " ").strip()[:120]
            print(f"  [{i}] id={h['chunk_id']:>4} {h['source'][:50]}")
            print(f"      {snippet}")
        print(f"  tokens: {result['usage']['total_tokens']}\n")


if __name__ == "__main__":
    main()
