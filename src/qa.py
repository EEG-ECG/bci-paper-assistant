"""
End-to-end RAG QA: retrieve top-k chunks → build prompt → call DeepSeek → return answer + citations.

Prompt 设计（关键）：
- System 严格限定"只能基于参考资料回答"，并要求用 [1][2][3] 形式标注引用来源——
  这是 RAG 减少幻觉的核心机制（强制模型把声明锚定到具体上下文）。
- 如果资料里没答案，明确要求模型说"资料中没有"，不要瞎编。
- 中英自动跟随用户问题的语言。

返回结构：{answer, hits, prompt}
- answer: LLM 文本
- hits: 检索到的 top-k chunks（含 source/score），用于UI显示和评测时核查
- prompt: 完整发给 LLM 的 messages（debug 用）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Windows UTF-8 控制台
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from openai import OpenAI

# 重用 Step 2 的 Retriever
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import Retriever

# 加载 .env
load_dotenv(Path(__file__).parent.parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

SYSTEM_PROMPT = """You are 「脑机接口论文知识助手」(a BCI / neuroscience paper assistant), built on a
knowledge base of 17 brain foundation model papers (BrainOmni, CBraMod, BrainWave, Brain-OF,
LaBraM, MindEye2, BrainBERT, TRIBE, etc.).

You will be given:
1. A user message (in Chinese or English).
2. Numbered reference passages [1], [2], [3], ... extracted from those academic papers.

== Conversational / meta messages (handle these FIRST) ==
If the user is merely greeting you (你好 / hi / hello), thanking you, saying goodbye, or asking
who you are / what you can do / how to use you — then IGNORE the reference passages and reply
briefly and warmly in the user's language. Introduce yourself as 「脑机接口论文知识助手」, say you
can answer questions about 17 brain foundation model papers (脑信号基础模型，覆盖 EEG/MEG/fMRI/iEEG),
and invite them to ask something specific (e.g. 数据集、benchmark 分数、模型架构、SOTA 对比).
Do NOT output the "not enough information" line for these conversational messages.

== Factual questions about the papers (the rest) ==
For any real question about the papers' content, follow these rules strictly:
- Answer ONLY using information present in the provided passages. Do not use prior knowledge.
- Cite the passages you use with bracket markers like [1] or [2] inline in your answer.
  If a sentence draws from multiple passages, cite them all, e.g. [1][3].
- If the passages do not contain enough information to answer the question, reply exactly:
  "提供的资料中没有足够信息回答这个问题。" (if the question is in Chinese)
  or "The provided passages do not contain enough information to answer this question." (if in English).
  Do not guess or fall back to general knowledge.
- Match the user's language: Chinese question → Chinese answer; English → English.
- Be concise and direct. Do not pad with restatements of the question.

== Handling linearized tables (IMPORTANT) ==
Academic PDFs often have tables that appear column-by-column or row-by-row as a vertical list.
When you see a sequence like:
    Header1
    Header2
    Header3
    Header4
    Value_A1
    Value_A2
    Value_A3
    Value_A4
    Value_B1
    Value_B2
    ...
recognize it as an N-column table (here N=4). Group consecutive values into rows of N, and
align each value to the corresponding header by position. Then you can extract specific cells
(e.g., "row B, column 3 = Value_B3"). It is OK to do this kind of structural inference — the
table is in the passage, you are reading it correctly, not guessing.

Similarly, Markdown-style tables `| col1 | col2 | col3 |` follow standard table semantics.

When extracting numeric values for specific (method, dataset, metric) combinations from results tables,
do this alignment carefully and cite the chunk that contains the table.
"""


def build_user_message(query: str, hits: list[dict]) -> str:
    """把 top-k chunks 拼成带编号的 reference 段。"""
    lines = ["Reference passages:\n"]
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {h['source']})")
        lines.append(h["text"].strip())
        lines.append("")
    lines.append(f"\nUser question: {query}")
    return "\n".join(lines)


class RAGSystem:
    def __init__(self, k: int = 10, mode: str = "rerank"):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        self.k = k
        self.mode = mode  # 'hybrid' | 'dense' | 'bm25'，用于 ablation
        self.retriever = Retriever()
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def answer(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        k = k if k is not None else self.k
        mode = mode if mode is not None else self.mode
        hits = self.retriever.search(query, k=k, mode=mode)
        user_msg = build_user_message(query, hits)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        resp = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer_text = resp.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "hits": hits,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            "model": DEEPSEEK_MODEL,
        }


def _print_result(result: dict, show_hits: bool = True):
    print("\n" + "=" * 80)
    print(f"Q: {result['query']}")
    print("-" * 80)
    print(f"A: {result['answer']}")
    if show_hits:
        print("\nRetrieved chunks (top-{}):".format(len(result["hits"])))
        for i, h in enumerate(result["hits"], 1):
            text = h["text"].replace("\n", " ")
            if len(text) > 150:
                text = text[:150] + "..."
            src = h["source"][:55]
            print(f"  [{i}] score={h['score']:.3f}  {src}")
            print(f"      {text}")
    print(f"\nTokens: prompt={result['usage']['prompt_tokens']}  "
          f"completion={result['usage']['completion_tokens']}  "
          f"total={result['usage']['total_tokens']}")


def smoke_test():
    rag = RAGSystem(k=5)
    queries = [
        "CBraMod 在多少个下游 BCI 任务和多少个公开数据集上做了评测？",
        "BrainOmni 预训练用了多少小时的 EEG 和 MEG 数据？",
        "What is the criss-cross transformer in CBraMod and how does it differ from full EEG modeling?",
        "Which brain foundation models support both EEG and MEG modalities?",
        # 故意问一个资料里没有的，验证模型会不会瞎编
        "Who is the president of France?",
    ]
    for q in queries:
        result = rag.answer(q)
        _print_result(result)


if __name__ == "__main__":
    smoke_test()
