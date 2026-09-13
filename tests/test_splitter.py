"""文本切分单元测试:保证块长上限、重叠行为与边界输入。"""

from __future__ import annotations

import pytest

from app.rag.splitter import normalize_text, split_text


def test_short_text_returns_single_chunk():
    chunks = split_text("这是一段很短的文档。", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "这是一段很短的文档。"
    assert chunks[0].index == 0


def test_empty_text_returns_no_chunk():
    assert split_text("   \n\n  ") == []


def test_every_chunk_respects_size_limit():
    text = "\n\n".join(f"第{i}段内容," + "测试文本" * 30 for i in range(10))
    chunks = split_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_oversized_paragraph_is_hard_split():
    text = "长" * 1000
    chunks = split_text(text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 6
    assert all(len(c.text) <= 150 for c in chunks)


def test_chunks_keep_position_metadata():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = split_text(text, chunk_size=30, chunk_overlap=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.end > chunk.start
        assert text[chunk.start:chunk.start + 4] in text


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, chunk_overlap=10)


def test_normalize_text_compresses_blank_lines():
    assert normalize_text("a\r\n\r\n\r\n\r\nb") == "a\n\nb"