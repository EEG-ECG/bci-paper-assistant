"""
Gradio web UI for the BCI-RAG system.

运行: python app.py  → 浏览器打开 http://127.0.0.1:7860

界面只是薄薄一层，真正逻辑在 src/qa.py 的 RAGSystem。
以后改 RAG（retrieve.py / qa.py）不需要动这个文件。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from qa import RAGSystem

# 启动时加载一次（模型加载较慢，约 20-40s）
print("Loading RAG system (BGE + reranker + ChromaDB + DeepSeek)... 约 20-40 秒")
RAG = RAGSystem(k=10, mode="rerank")
print("RAG system ready.")

EXAMPLES = [
    "CBraMod 在多少个下游 BCI 任务和多少个公开数据集上做了评测？",
    "BrainOmni 预训练用了多少小时的 EEG 和 MEG 数据？",
    "What is the criss-cross transformer in CBraMod and how does it differ from full EEG modeling?",
    "BrainWave 论文中 ADHD-Adult 数据集有多少样本？",
    "哪些 brain foundation model 同时支持 EEG 和 MEG？",
]


def answer_fn(question: str, k: int, mode: str):
    question = (question or "").strip()
    if not question:
        return "请输入问题。", "", ""

    result = RAG.answer(question, k=int(k), mode=mode)

    answer_md = result["answer"]

    # 引用来源
    cites = []
    for i, h in enumerate(result["hits"], 1):
        src = h["source"]
        score = h.get("score", 0.0)
        stype = h.get("score_type", mode)
        text = h["text"].strip()
        if len(text) > 600:
            text = text[:600] + " …"
        is_table = "📊 表格" if h.get("doc_chunk_id", 0) == -1 else ""
        cites.append(
            f"**[{i}]** `{src}` · score={score:.3f} ({stype}) {is_table}\n\n"
            f"```\n{text}\n```"
        )
    citations_md = "\n\n---\n\n".join(cites) if cites else "（无）"

    u = result["usage"]
    meta = (
        f"🤖 {result['model']}  |  检索: {mode}, top-k={int(k)}  |  "
        f"tokens: prompt={u['prompt_tokens']} + completion={u['completion_tokens']} = {u['total_tokens']}"
    )
    return answer_md, citations_md, meta


with gr.Blocks(title="BCI-RAG 脑机接口论文问答") as demo:
    gr.Markdown(
        "# 🧠 BCI-RAG：脑机接口领域论文问答系统\n"
        "基于 17 篇 brain foundation model 论文（BrainOmni / CBraMod / BrainWave / Brain-OF 等），"
        "用 RAG（检索增强生成）回答专业问题，并给出原文引用。\n\n"
        "**技术栈**：PyMuPDF 解析 · BGE embedding · ChromaDB · BM25+Dense hybrid 检索 · "
        "BGE-reranker 重排 · DeepSeek-V4 生成"
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="你的问题",
                placeholder="例如：CBraMod 在 BCIC2020-3 上的 Balanced Accuracy 是多少？",
                lines=2,
            )
            with gr.Row():
                submit = gr.Button("提问", variant="primary")
                clear = gr.Button("清空")
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["rerank", "hybrid", "dense", "bm25"],
                value="rerank",
                label="检索模式",
                info="rerank=两阶段(最好) / hybrid=BM25+向量 / dense=纯向量 / bm25=纯关键词",
            )
            k = gr.Slider(
                minimum=3, maximum=20, value=10, step=1,
                label="top-k（给 LLM 几个 chunk）",
            )

    meta = gr.Markdown("")

    # 注：早期版本把答案放进 gr.Tab 里，Gradio 6.x 下 Tab 内的 Markdown 更新后不渲染
    # （后端返回正常但页面空白）。改成直接可见的 Markdown + 可折叠 Accordion 即正常。
    gr.Markdown("### 答案")
    answer_out = gr.Markdown(value="*在上方输入问题并点击「提问」*")

    with gr.Accordion("📚 引用来源（检索到的 chunks）", open=False):
        citations_out = gr.Markdown(value="")

    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[question])

    submit.click(
        answer_fn, inputs=[question, k, mode], outputs=[answer_out, citations_out, meta]
    )
    question.submit(
        answer_fn, inputs=[question, k, mode], outputs=[answer_out, citations_out, meta]
    )
    clear.click(
        lambda: ("", "*在上方输入问题并点击「提问」*", "", ""),
        outputs=[question, answer_out, citations_out, meta],
    )


if __name__ == "__main__":
    # share=True 生成公网链接 (xxxxx.gradio.live)，72小时有效，电脑须保持运行。
    # auth 加访问密码：只有知道账号密码的人能用，防止链接外泄被滥刷 API。
    #   ↓↓↓ 改成你自己的账号/密码 ↓↓↓
    AUTH = ("bci", "change-me-2026")
    demo.launch(
        share=True,
        auth=AUTH,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
