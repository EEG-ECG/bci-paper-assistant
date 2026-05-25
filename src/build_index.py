"""
Build vector index: embed all chunks with BGE and store in ChromaDB.

关键设计：
- 模型：BAAI/bge-base-en-v1.5 (768 dim, ~440MB)。BGE 是 BAAI 自家的检索模型，
  在 MTEB 等检索 benchmark 上表现强于 all-MiniLM 之类的早期模型。
- 文档侧：直接 encode chunk 文本，不加前缀（BGE 训练时 passage 无前缀）。
- normalize_embeddings=True：BGE 期望归一化向量，配合 cosine 相似度使用。
- ChromaDB 持久化到 ./chroma_db/，下次启动直接 PersistentClient 重连即可。
- 用 HF 镜像 (hf-mirror.com)：huggingface.co 在国内不稳定。
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")
CHROMA_DIR = Path("E:/RAG_C/chroma_db")
MODEL_NAME = "BAAI/bge-base-en-v1.5"  # 从 ModelScope 下载（HuggingFace 在国内连不上）
COLLECTION_NAME = "bci_papers"
EMBED_BATCH = 16  # CPU embedding，batch 别太大
CHROMA_ADD_BATCH = 100


def load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    ]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Downloading {MODEL_NAME} from ModelScope (~440MB first time)...")
    model_dir = snapshot_download(MODEL_NAME)
    print(f"  model dir: {model_dir}")
    print("Loading SentenceTransformer from local dir...")
    model = SentenceTransformer(model_dir)
    print(f"  embedding dim = {model.get_sentence_embedding_dimension()}")

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  embeddings shape: {embeddings.shape}")

    print(f"Storing in ChromaDB at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 幂等：删了重建
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  removed existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    coll = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": MODEL_NAME},
    )

    # 分批写入：ChromaDB 单批有上限
    for i in tqdm(range(0, len(chunks), CHROMA_ADD_BATCH), desc="Adding to Chroma"):
        batch = chunks[i : i + CHROMA_ADD_BATCH]
        coll.add(
            ids=[str(c["chunk_id"]) for c in batch],
            embeddings=embeddings[i : i + CHROMA_ADD_BATCH].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "doc_chunk_id": c["doc_chunk_id"],
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                }
                for c in batch
            ],
        )

    print(f"\nIndexed {coll.count()} chunks into collection '{COLLECTION_NAME}'")
    print(f"Persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
