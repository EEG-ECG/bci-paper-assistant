"""
Build BM25 index over all chunks, save to data/bm25_index.pkl.

为什么需要 BM25：
- dense embedding (BGE) 对"精确编号/术语"匹配能力差。
  例如 "BCIC2020-3"、"PD-31"、"ADHD-200" 这类数据集编号，embedding 把它们当
  普通数字串，loss 中没有强信号区分。
- BM25 是 1990 年代的传统 IR 算法，对 term frequency 敏感，精确 token 匹配很强。
- Hybrid (BM25 + dense) 是工业 RAG 的事实标准，覆盖两类失败模式：
  BM25 → 精确术语；dense → 语义/同义/解释类。

Tokenization 设计（关键）：
- 保留连字符 token：'BCIC2020-3' / 'ADHD-200' 作为整体 token，不切散。
- 小写化。
- 不分中文（因为 corpus 是英文论文，中文查询里的中文部分本来就无法 BM25 命中；
  我们靠中文查询里的英文术语和数字命中 chunks）。
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

CHUNKS_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")
OUT_PATH = Path("E:/RAG_C/data/bm25_index.pkl")

# 正则：保留含连字符的 alphanumeric token
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_/]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def main():
    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    ]
    print(f"Loaded {len(chunks)} chunks")

    print("Tokenizing...")
    tokenized = [tokenize(c["text"]) for c in tqdm(chunks)]
    chunk_ids = [c["chunk_id"] for c in chunks]

    n_tokens = sum(len(t) for t in tokenized)
    print(f"  total tokens: {n_tokens:,}, avg per chunk: {n_tokens / len(chunks):.1f}")

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved BM25 index to {OUT_PATH} ({size_mb:.1f} MB)")

    # 抽样查询验证
    print("\n=== Quick sanity check ===")
    queries = ["BCIC2020-3", "PD-31 dataset", "ADHD-Adult sample"]
    for q in queries:
        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        top = sorted(zip(chunk_ids, scores), key=lambda x: -x[1])[:3]
        print(f"\n  Q: '{q}' (tokens: {q_tokens})")
        for cid, sc in top:
            text = chunks[cid]["text"].replace("\n", " ")[:80]
            print(f"    bm25={sc:.2f}  chunk_id={cid}  '{text}...'")


if __name__ == "__main__":
    main()
