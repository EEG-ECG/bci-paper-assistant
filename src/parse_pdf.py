"""
Parse academic PDFs into clean text (PyMuPDF version).

为什么用 PyMuPDF (fitz) 而不是 pdfplumber：
- fitz 的 get_text() 默认就能正确处理双栏布局的阅读顺序，无需手写列检测。
- fitz 解析出的字符间距正确，不会出现 'JiquanWang' 这种粘连。
- fitz 通过 get_text("dict") 还能拿到每个 span 的字体大小，可以用字号特征
  精准检测 References 标题（往往字号偏大），比正则匹配可靠得多。

流程：
1. 逐页 get_text()，按阅读顺序拿到文本。
2. 用 get_text("dict") 找全文最大字号的几个候选标题，定位 "References" 类标题
   在原文中的位置，截断之后丢弃。
3. 过滤多页重复的页眉页脚短行。
4. 合并连字符断行、压缩多余空行。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

ARTICLE_DIR = Path("E:/RAG_C/Article")
OUTPUT_DIR = Path("E:/RAG_C/data/raw_text")

REF_PATTERNS = re.compile(
    r"^(?:\d+\.?\s+)?(references|bibliography|reference)\s*$",
    re.IGNORECASE,
)


def extract_text_per_page(doc: fitz.Document) -> list[str]:
    """每页用默认 get_text()，让 fitz 自己处理双栏阅读顺序。"""
    return [page.get_text() for page in doc]


def find_references_cutoff(doc: fitz.Document) -> tuple[int, int] | None:
    """用字号特征 + 文本匹配定位 References 标题。

    思路：扫所有 span，找文本是 References/Bibliography 且字号 >= 全文 P90 字号
    （或与"1 Introduction" 这类章节标题字号一致）的 span。
    返回 (page_idx, y_position)，调用方据此截断。
    """
    # 第一步：统计所有 span 的字号分布
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        sizes.append(span["size"])
    if not sizes:
        return None
    sizes.sort()
    # 标题字号阈值：取 P85
    threshold = sizes[int(len(sizes) * 0.85)]

    for page_idx, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line.get("spans", []))
                if not text.strip():
                    continue
                if not REF_PATTERNS.match(text.strip()):
                    continue
                # 字号判断：行内任一 span 字号 >= 阈值
                max_size = max((s["size"] for s in line.get("spans", [])), default=0)
                if max_size >= threshold:
                    y = line["bbox"][1]
                    return (page_idx, y)
    return None


def apply_references_cutoff(pages: list[str], doc: fitz.Document, cutoff) -> list[str]:
    if cutoff is None:
        return pages
    page_idx, y = cutoff
    # 截断 page_idx 这一页：只保留 y 之前的内容；之后的页全部丢
    kept = []
    for i, page_text in enumerate(pages):
        if i < page_idx:
            kept.append(page_text)
        elif i == page_idx:
            # 用 fitz 重新提取该页 y 之前的文本
            page = doc[i]
            clip_rect = fitz.Rect(0, 0, page.rect.width, y)
            kept.append(page.get_text(clip=clip_rect))
        # i > page_idx: 丢弃
    return kept


def detect_repeated_headers_footers(pages_text: list[str]) -> set[str]:
    counter: Counter[str] = Counter()
    for text in pages_text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        candidates = lines[:2] + lines[-2:]
        for ln in candidates:
            if 5 <= len(ln) <= 100:
                counter[ln] += 1
    threshold = max(2, int(len(pages_text) * 0.3))
    return {ln for ln, c in counter.items() if c >= threshold}


def clean_text(text: str) -> str:
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text.strip()


def parse_pdf(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as doc:
        pages = extract_text_per_page(doc)
        # 注：曾在此处按 References 标题截断后续内容，但发现学术 PDF 常有
        # body → References → Appendix → 关键 result tables 的结构。
        # 截断会切掉 30-70% 的实际数据（如 CBraMod 的 BCIC2020-3 results 表）。
        # 改为保留全文。参考文献条目会成为低召回率的噪声 chunks，
        # 实际查询不太会命中，影响可接受。

    headers_footers = detect_repeated_headers_footers(pages)
    cleaned_pages = []
    for page_text in pages:
        kept_lines = [
            ln for ln in page_text.splitlines() if ln.strip() not in headers_footers
        ]
        cleaned_pages.append("\n".join(kept_lines))

    full_text = "\n\n".join(cleaned_pages)
    return clean_text(full_text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(ARTICLE_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {ARTICLE_DIR}")

    stats = []
    for pdf_path in tqdm(pdfs, desc="Parsing"):
        try:
            text = parse_pdf(pdf_path)
            out_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            stats.append((pdf_path.name, len(text)))
        except Exception as e:
            print(f"[FAILED] {pdf_path.name}: {e}")
            stats.append((pdf_path.name, -1))

    print("\n=== Parsing summary ===")
    for name, n in stats:
        s = f"{n:>8} chars" if n >= 0 else "  FAILED"
        print(f"  {s}  {name}")
    total = sum(n for _, n in stats if n >= 0)
    print(f"\nTotal: {total:,} chars across {sum(1 for _, n in stats if n >= 0)} files")


if __name__ == "__main__":
    main()
