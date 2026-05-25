"""Sample N random chunks and dump them to a readable file for human QA."""

from __future__ import annotations

import json
import random
from pathlib import Path

CHUNKS_PATH = Path("E:/RAG_C/data/chunks/chunks.jsonl")
OUT_PATH = Path("E:/RAG_C/data/chunks/sample_for_qa.txt")
N = 20
SEED = 42


def main():
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    random.seed(SEED)
    sample = random.sample(chunks, N)

    lines = []
    for c in sample:
        lines.append("=" * 80)
        lines.append(f"chunk_id={c['chunk_id']}  doc_chunk={c['doc_chunk_id']}  n_chars={c['n_chars']}")
        lines.append(f"source: {c['source']}")
        lines.append(f"char_range: [{c['char_start']}, {c['char_end']})")
        lines.append("-" * 80)
        lines.append(c["text"])
        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {N} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
