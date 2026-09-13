"""知识库服务测试:文档加载、入库与检索包装。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.embedder import HashingNgramEmbedder
from app.rag.loader import UnsupportedFileType, load_bytes
from app.rag.store import VectorStore
from app.services.kb_service import KBService


def _service(work_dir: Path) -> KBService:
    store = VectorStore(work_dir / "index", HashingNgramEmbedder(dim=256))
    return KBService(store, chunk_size=120, chunk_overlap=20)


def test_add_text_and_search(work_dir):
    kb = _service(work_dir)
    kb.add_text("退货政策:商品签收后 7 天内可无理由退货,需保持包装完整。", title="售后政策")
    hits = kb.search("退货要多久之内", top_k=1)
    assert hits and hits[0].title == "售后政策"


def test_add_file_supports_markdown(work_dir):
    kb = _service(work_dir)
    record = kb.add_file("readme.md", "# 标题\n\n这是 Markdown 内容。".encode("utf-8"))
    assert record.title == "readme.md"
    assert record.chunk_count >= 1


def test_add_empty_text_raises(work_dir):
    kb = _service(work_dir)
    with pytest.raises(ValueError):
        kb.add_text("   ", title="空文档")


def test_unsupported_file_type_raises():
    with pytest.raises(UnsupportedFileType):
        load_bytes("demo.xlsx", b"whatever")