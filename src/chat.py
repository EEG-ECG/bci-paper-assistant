"""
Interactive REPL for the RAG system.

用法：python src/chat.py
- 输入问题回车 → 得到答案 + 引用的 chunks
- 输入 'q' / 'quit' / 'exit' 或 Ctrl+C 退出
- 输入 'k=N' 临时调整 top-k（如 'k=20'）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 关掉 ChromaDB 的 telemetry，避免启动时初始化 OpenTelemetry（慢且没意义）
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# 提前 print，避免黑屏几秒以为卡死了。flush=True 立即输出。
print("[1/3] Loading dependencies (this can take 10-20s)...", flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from qa import RAGSystem
print("[2/3] Dependencies loaded. Initializing models...", flush=True)


def main():
    rag = RAGSystem(k=10, mode="rerank")
    print("[3/3] Ready. Type your question and press Enter.")
    print("Commands: 'mode=dense|bm25|hybrid|rerank' to switch retrieval mode")
    print("Commands: 'q' / 'quit' to exit, 'k=N' to change top-k (e.g. 'k=20').")
    print("-" * 80)

    current_k = 20
    current_mode = "hybrid"
    while True:
        try:
            q = input("\n你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break
        if q.lower().startswith("k="):
            try:
                current_k = int(q.split("=", 1)[1])
                print(f"  top-k set to {current_k}")
            except ValueError:
                print("  invalid k value")
            continue
        if q.lower().startswith("mode="):
            m = q.split("=", 1)[1].strip().lower()
            if m in ("hybrid", "dense", "bm25"):
                current_mode = m
                print(f"  mode set to {current_mode}")
            else:
                print("  invalid mode (use hybrid|dense|bm25)")
            continue

        try:
            result = rag.answer(q, k=current_k, mode=current_mode)
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue

        print("\n" + "=" * 80)
        print("答案:")
        print(result["answer"])
        print("\n" + "-" * 80)
        print(f"引用来源 (top-{current_k}, mode={current_mode}):")
        for i, h in enumerate(result["hits"], 1):
            snippet = h["text"].replace("\n", " ").strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            src = h["source"]
            print(f"  [{i}] score={h['score']:.3f}  {src[:70]}")
            print(f"      {snippet}")
        u = result["usage"]
        print(
            f"\nTokens: prompt={u['prompt_tokens']}  "
            f"completion={u['completion_tokens']}  total={u['total_tokens']}"
        )


if __name__ == "__main__":
    main()
