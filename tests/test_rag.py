"""向量化与向量库测试:覆盖入库、检索命中、持久化与删除。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.rag.embedder import HashingNgramEmbedder, build_embedder
from app.rag.splitter import split_text
from app.rag.store import VectorStore

DOCS = [
    ("Python 基础", "Python 是一种解释型语言。列表是一种有序可变序列,用中括号定义,支持增删改查。"),
    ("数据库索引", "数据库索引可以加快查询速度。MySQL 的 InnoDB 使用 B+ 树实现聚簇索引。"),
    ("Docker 入门", "Docker 用容器隔离运行环境。Dockerfile 描述镜像构建步骤,镜像可以分层缓存。"),
]


def _build_store(work_dir: Path) -> VectorStore:
    store = VectorStore(work_dir / "index", HashingNgramEmbedder(dim=512))
    for title, text in DOCS:
        store.add_document(title=title, source=title, chunks=split_text(text, 200, 40))
    return store


def test_factory_returns_hash_embedder_by_default():
    assert isinstance(build_embedder("hash", 256), HashingNgramEmbedder)


def test_embeddings_are_normalized():
    embedder = HashingNgramEmbedder(dim=256)
    embedder.fit([d[1] for d in DOCS])
    vectors = embedder.embed_documents([d[1] for d in DOCS])
    assert vectors.shape == (3, 256)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_search_returns_relevant_document(work_dir):
    store = _build_store(work_dir)
    hits = store.search("列表怎么用", top_k=3)
    assert hits, "应当召回结果"
    assert "Python" in hits[0].title or "列表" in hits[0].text


def test_search_returns_empty_for_blank_query(work_dir):
    store = _build_store(work_dir)
    assert store.search("   ") == []


def test_threshold_filters_low_score_hits(work_dir):
    store = _build_store(work_dir)
    assert store.search("完全无关的量子物理话题", top_k=3, threshold=0.35) == []


def test_index_is_persisted_and_reloadable(work_dir):
    store = _build_store(work_dir)
    reloaded = VectorStore(work_dir / "index", HashingNgramEmbedder(dim=512)).load()
    assert reloaded.stats()["documents"] == 3
    assert reloaded.stats()["chunks"] == store.stats()["chunks"]
    before = store.search("Docker 镜像", top_k=1)[0].title
    after = reloaded.search("Docker 镜像", top_k=1)[0].title
    assert before == after


def test_delete_document_removes_chunks(work_dir):
    store = _build_store(work_dir)
    doc_id = store.list_documents()[0].doc_id
    assert store.delete_document(doc_id) is True
    assert store.delete_document(doc_id) is False
    assert store.stats()["documents"] == 2
    assert all(c.doc_id != doc_id for c in store.chunks)