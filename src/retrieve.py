"""
Retrieve top-k chunks for a query. Supports dense / BM25 / hybrid / rerank modes.

两阶段检索 (默认 mode='rerank')：
  阶段1 召回 (recall)：hybrid (BM25 + dense via RRF) 取 top-RERANK_POOL 个候选。
    目标是高召回——把可能相关的都捞进来，不怕噪声。
  阶段2 重排 (rerank)：用 cross-encoder (BGE-reranker) 对每个 (query, passage) 对
    精细打分，取 top-k。cross-encoder 让 query 和 passage 在同一个 transformer 里做
    full attention，比 bi-encoder (dense) 的"各自编码再算余弦"精确得多，尤其擅长
    判断"这段到底有没有回答这个问题"，能把含具体数字/事实的 chunk 顶上来。

为什么需要 hybrid（阶段1）：
- dense (BGE) 对语义/同义/解释类查询强，但精确编号 (BCIC2020-3, PD-31) 召回差。
- BM25 对精确 term 召回强，但语义匹配能力弱（同义词、改写、跨语言完全无效）。
- Hybrid (RRF) 取长补短，是工业 RAG 的事实标准。

为什么需要 rerank（阶段2）：
- bi-encoder 召回的 top-N 里，真正含答案的 chunk 常被"宽泛但相关"的 chunk 挤到
  rank 20+。cross-encoder 重排能把它顶回 top-k。
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from modelscope import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from build_bm25 import tokenize  # 用同一套 tokenizer 保证 build/query 一致

CHROMA_DIR = Path("E:/RAG_C/chroma_db")
BM25_PATH = Path("E:/RAG_C/data/bm25_index.pkl")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-base"  # cross-encoder，支持中英
COLLECTION_NAME = "bci_papers"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RRF_K = 60  # RRF 经验常数
CANDIDATE_POOL = 60  # 每种检索器各取 top-N 进入融合池
# rerank 模式下，阶段1 hybrid 召回多少候选交给 reranker。
# cross-encoder 在 CPU 上每个候选 ~0.6s，池子越大越慢。
# 实测关键 chunk 基本都在 hybrid top-20 内，25 足够，兼顾速度与精度。
RERANK_POOL = 25


class Retriever:
    def __init__(self, use_reranker: bool = True):
        # Dense
        model_dir = snapshot_download(MODEL_NAME)
        self.model = SentenceTransformer(model_dir)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.coll = client.get_collection(COLLECTION_NAME)

        # BM25
        with BM25_PATH.open("rb") as f:
            obj = pickle.load(f)
        self.bm25 = obj["bm25"]
        self.bm25_chunk_ids = obj["chunk_ids"]  # ordered chunk_ids in BM25 index
        self.id_to_bm25_idx = {cid: i for i, cid in enumerate(self.bm25_chunk_ids)}

        # Reranker (cross-encoder)
        self.reranker = None
        if use_reranker:
            reranker_dir = snapshot_download(RERANKER_NAME)
            self.reranker = CrossEncoder(reranker_dir)

        print(
            f"Retriever ready. dense={self.coll.count()} chunks, "
            f"bm25={len(self.bm25_chunk_ids)} chunks, "
            f"reranker={'on' if self.reranker else 'off'}"
        )

    # ---------- low-level: dense / bm25 each ----------
    def _dense_topn(self, query: str, n: int) -> list[tuple[int, float]]:
        """Returns [(chunk_id, dense_score)] sorted by score desc."""
        q_emb = self.model.encode(
            QUERY_PREFIX + query, normalize_embeddings=True
        ).tolist()
        res = self.coll.query(query_embeddings=[q_emb], n_results=n)
        ids = [int(x) for x in res["ids"][0]]
        scores = [1 - d for d in res["distances"][0]]  # cosine sim
        return list(zip(ids, scores))

    def _bm25_topn(self, query: str, n: int) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # 取 top-n
        idx_score = sorted(enumerate(scores), key=lambda x: -x[1])[:n]
        return [(self.bm25_chunk_ids[i], s) for i, s in idx_score]

    # ---------- public: search ----------
    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "rerank",
    ) -> list[dict]:
        """mode: 'dense' | 'bm25' | 'hybrid' | 'rerank'"""
        if mode == "dense":
            hits = self._dense_topn(query, k)
        elif mode == "bm25":
            hits = self._bm25_topn(query, k)
        elif mode == "hybrid":
            hits = self._rrf_hybrid(query, k)
        elif mode == "rerank":
            if self.reranker is None:
                raise RuntimeError("reranker not loaded; init Retriever(use_reranker=True)")
            hits = self._rerank(query, k)
        else:
            raise ValueError(f"unknown mode: {mode}")

        return self._materialize(hits, query, mode)

    def _rerank(self, query: str, k: int) -> list[tuple[int, float]]:
        """阶段1 hybrid 召回 RERANK_POOL 候选 → 阶段2 cross-encoder 重排 → top-k。"""
        pool = self._rrf_hybrid(query, RERANK_POOL)
        cand_ids = [cid for cid, _ in pool]
        if not cand_ids:
            return []

        # 取候选文本
        res = self.coll.get(ids=[str(c) for c in cand_ids], include=["documents"])
        id_to_text = {int(res["ids"][i]): res["documents"][i] for i in range(len(res["ids"]))}

        pairs = [(query, id_to_text.get(cid, "")) for cid in cand_ids]
        scores = self.reranker.predict(pairs)  # higher = more relevant

        ranked = sorted(zip(cand_ids, scores), key=lambda x: -float(x[1]))[:k]
        return [(cid, float(s)) for cid, s in ranked]

    def _rrf_hybrid(self, query: str, k: int) -> list[tuple[int, float]]:
        dense_results = self._dense_topn(query, CANDIDATE_POOL)
        bm25_results = self._bm25_topn(query, CANDIDATE_POOL)

        scores: dict[int, float] = {}
        for rank, (cid, _) in enumerate(dense_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (cid, _) in enumerate(bm25_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return top

    def _materialize(
        self, hits: list[tuple[int, float]], query: str, mode: str
    ) -> list[dict]:
        if not hits:
            return []
        ids = [str(cid) for cid, _ in hits]
        res = self.coll.get(ids=ids, include=["documents", "metadatas"])
        # ChromaDB get() doesn't preserve order — rebuild via dict
        id_to_data = {}
        for i in range(len(res["ids"])):
            id_to_data[int(res["ids"][i])] = {
                "text": res["documents"][i],
                "meta": res["metadatas"][i],
            }

        out = []
        for cid, score in hits:
            d = id_to_data.get(cid)
            if not d:
                continue
            out.append(
                {
                    "chunk_id": cid,
                    "score": round(float(score), 4),
                    "score_type": "rrf" if mode == "hybrid" else mode,
                    "source": d["meta"]["source"],
                    "doc_chunk_id": d["meta"]["doc_chunk_id"],
                    "text": d["text"],
                }
            )
        return out


def _fmt(h: dict, max_text: int = 130) -> str:
    text = h["text"].replace("\n", " ").strip()
    if len(text) > max_text:
        text = text[:max_text] + "..."
    return (
        f"  [score={h['score']:.4f} ({h['score_type']}) id={h['chunk_id']:>4}] "
        f"{h['source'][:55]}\n    {text}"
    )


def smoke_test():
    r = Retriever()
    queries = [
        "CBraMod 在 BCIC2020-3 数据集上的 Balanced Accuracy 是多少?",
        "PD-31 数据集有多少被试?",
        "ADHD-200 用什么脑图谱?",
    ]
    for q in queries:
        for mode in ["dense", "bm25", "hybrid"]:
            print("\n" + "=" * 80)
            print(f"Q: {q}    [mode={mode}]")
            print("-" * 80)
            for h in r.search(q, k=3, mode=mode):
                print(_fmt(h))


if __name__ == "__main__":
    smoke_test()
