"""文本切分:按段落聚合、超长段落硬切、块间保留重叠。

切分质量直接决定召回质量,所以这里不做简单的定长切片,而是:
    1. 先按空行切段落,尽量保持语义完整;
    2. 段落按顺序装进不超过 chunk_size 的块里;
    3. 单段超长时按 chunk_size 硬切,并用 chunk_overlap 保留上下文;
    4. 相邻块之间拼接上一块的尾部,避免答案正好落在切口上。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """一个文本块,保留在原文中的位置便于引用溯源。"""

    text: str
    index: int
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    """统一换行、压缩多余空白,避免同一文档因排版差异产生不同向量。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """把超长段落切成带重叠的小块。"""
    step = max(1, size - overlap)
    pieces: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


def split_text(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> list[Chunk]:
    """把长文本切成若干带位置信息的块。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

    blocks: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            blocks.append(para)
        else:
            blocks.extend(_hard_split(para, chunk_size, chunk_overlap))

    chunks: list[Chunk] = []
    buffer = ""
    cursor = 0

    def flush() -> None:
        nonlocal buffer, cursor
        if not buffer:
            return
        start = normalized.find(buffer, cursor)
        if start < 0:                      # 理论上不会发生,兜底避免位置错乱
            start = cursor
        chunks.append(Chunk(text=buffer, index=len(chunks), start=start, end=start + len(buffer)))
        cursor = start + len(buffer)
        buffer = ""

    for block in blocks:
        if not buffer:
            buffer = block
            continue
        if len(buffer) + 1 + len(block) <= chunk_size:
            buffer = f"{buffer}\n{block}"
            continue
        tail = buffer[-chunk_overlap:] if chunk_overlap else ""
        flush()
        # 重叠部分只有在放得下时才拼接,保证每个块都不超过 chunk_size
        if tail and len(tail) + 1 + len(block) <= chunk_size:
            buffer = f"{tail}\n{block}"
            cursor = max(0, cursor - len(tail))
        else:
            buffer = block

    flush()
    return chunks