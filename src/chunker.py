"""
Split parsed text into chunks with metadata.

切分策略：
- RecursiveCharacterTextSplitter：按 [\n\n, \n, ". ", " ", ""] 优先级递归切分，
  优先在段落、句子边界处切，避免把一句话切两半。
- chunk_size=600, chunk_overlap=80：学术正文段落偏长，600 字符约 100-150 词，
  相邻 chunk 重叠 80 字符防关键句被截。
- 元数据：source（来源论文）、chunk_id（全局）、doc_chunk_id（文档内序号）、
  char_start/char_end（在原文中的字符位置，便于回溯）。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

RAW_DIR = Path("E:/RAG_C/data/raw_text")
OUT_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")
STATS_PATH = Path("E:/RAG_C/data/chunks/stats.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def find_position(haystack: str, needle: str, start: int = 0) -> int:
    """因为 splitter 可能 strip 空白，直接 find 可能找不到；用第一段非空字符近似定位。"""
    if not needle.strip():
        return start
    probe = needle.strip()[:60]
    idx = haystack.find(probe, start)
    return idx if idx >= 0 else start


def chunk_one_doc(text: str, source: str, splitter, start_chunk_id: int) -> list[dict]:
    pieces = splitter.split_text(text)
    chunks = []
    cursor = 0
    for i, piece in enumerate(pieces):
        start = find_position(text, piece, cursor)
        end = start + len(piece)
        chunks.append(
            {
                "chunk_id": start_chunk_id + i,
                "doc_chunk_id": i,
                "source": source,
                "char_start": start,
                "char_end": end,
                "n_chars": len(piece),
                "text": piece,
            }
        )
        cursor = max(0, end - CHUNK_OVERLAP)
    return chunks


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splitter = make_splitter()

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {RAW_DIR}")

    all_chunks = []
    per_doc_counts = {}
    for txt_path in tqdm(txt_files, desc="Chunking"):
        text = txt_path.read_text(encoding="utf-8")
        source = txt_path.stem
        chunks = chunk_one_doc(text, source, splitter, start_chunk_id=len(all_chunks))
        all_chunks.extend(chunks)
        per_doc_counts[source] = len(chunks)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 汇总统计
    sizes = [c["n_chars"] for c in all_chunks]
    stats = {
        "chunk_size_target": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": len(all_chunks),
        "total_docs": len(txt_files),
        "size_min": min(sizes),
        "size_max": max(sizes),
        "size_mean": round(sum(sizes) / len(sizes), 1),
        "per_doc_chunks": per_doc_counts,
    }
    STATS_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {len(all_chunks):,} chunks to {OUT_PATH}")
    print(f"Chunk size: min={stats['size_min']}, max={stats['size_max']}, mean={stats['size_mean']}")
    print("Per-doc counts:")
    for src, n in sorted(per_doc_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {src[:70]}")


if __name__ == "__main__":
    main()
