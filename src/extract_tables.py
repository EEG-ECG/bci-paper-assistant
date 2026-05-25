"""
Extract tables from PDFs using PyMuPDF's page.find_tables() and add as structured chunks.

为什么需要这个：
- 学术 PDF 中 "Method × Dataset → Metric" 这种表格，用 extract_text() 提取后会被
  线性化（每个 cell 变成单独一行），行列对应关系丢失，LLM 无法跨表读出 (X,Y) 单元格的值。
- fitz.Page.find_tables() 用基于线段/对齐的启发式检测表格区域，可以返回结构化的
  rows-of-cells，让我们重建 Markdown 风格的表格文本（保留行列对应）。
- 把每张表作为一个独立 chunk 加进检索池，配合 hybrid (BM25+dense) 能让"在 X 数据集
  上 Y 方法的分数"这类问题命中正确的表。

输出：覆盖 data/chunks/chunks.jsonl，保留原文本 chunks（type=text），追加表格
chunks（type=table）。
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
from tqdm import tqdm

ARTICLE_DIR = Path("E:/RAG_C/Article")
CHUNKS_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")

MAX_TABLE_TEXT = 3000  # 超过截断（BGE 输入 token 限制 ~512）
MIN_ROWS = 2  # 少于 2 行（一个 header + 一行数据）的"表格"忽略
MIN_USEFUL_CHARS = 50


def format_table(
    rows: list[list[str | None]], source: str, page_idx: int, table_idx: int
) -> str:
    """Render rows as Markdown-ish table with explicit header marker."""
    header = [(c or "").strip() for c in rows[0]]
    n_cols = len(header)
    data_rows = rows[1:]

    lines = [
        f"[Table {table_idx + 1} from paper: {source}, page {page_idx + 1}]",
        " | ".join(header) if any(header) else "(no header)",
        " | ".join(["---"] * n_cols),
    ]
    for row in data_rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        # 填充/截断到 header 列数
        if len(cells) < n_cols:
            cells = cells + [""] * (n_cols - len(cells))
        elif len(cells) > n_cols:
            cells = cells[:n_cols]
        lines.append(" | ".join(cells))

    text = "\n".join(lines)
    if len(text) > MAX_TABLE_TEXT:
        text = text[:MAX_TABLE_TEXT] + "\n... [table truncated]"
    return text


def main():
    # 读现有 chunks，剔除任何旧的 table chunks（避免重复追加）
    existing = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    ]
    text_chunks = [c for c in existing if c.get("type", "text") == "text"]
    next_id = max(c["chunk_id"] for c in text_chunks) + 1 if text_chunks else 0
    print(f"Loaded {len(text_chunks)} text chunks. Next id starts at {next_id}.")

    table_chunks = []
    n_total_tables = 0
    n_kept_tables = 0

    pdfs = sorted(ARTICLE_DIR.glob("*.pdf"))
    for pdf_path in tqdm(pdfs, desc="Extracting tables"):
        source = pdf_path.stem
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"\n[FAILED to open] {pdf_path.name}: {e}")
            continue
        try:
            for page_idx, page in enumerate(doc):
                try:
                    tables = page.find_tables()
                except Exception:
                    continue
                for table_idx, table in enumerate(tables.tables):
                    n_total_tables += 1
                    try:
                        rows = table.extract()
                    except Exception:
                        continue
                    if not rows or len(rows) < MIN_ROWS:
                        continue
                    text = format_table(rows, source, page_idx, table_idx)
                    if len(text.strip()) < MIN_USEFUL_CHARS:
                        continue

                    table_chunks.append(
                        {
                            "chunk_id": next_id,
                            "doc_chunk_id": -1,
                            "source": source,
                            "char_start": -1,
                            "char_end": -1,
                            "n_chars": len(text),
                            "text": text,
                            "type": "table",
                            "page": page_idx,
                        }
                    )
                    next_id += 1
                    n_kept_tables += 1
        finally:
            doc.close()

    print(f"\nDetected {n_total_tables} tables total, kept {n_kept_tables} (>= {MIN_ROWS} rows).")

    # 重写 chunks.jsonl：原 text chunks + 新 table chunks
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in text_chunks + table_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(text_chunks) + len(table_chunks)} total chunks to {CHUNKS_PATH}")
    print(f"  text chunks: {len(text_chunks)}")
    print(f"  table chunks: {len(table_chunks)}")

    # 抽样看几个表格
    print("\n=== Sample tables ===")
    for c in table_chunks[:3]:
        print(f"\n--- chunk {c['chunk_id']} ({c['source'][:50]}, page {c['page']}) ---")
        print(c["text"][:500])


if __name__ == "__main__":
    main()
