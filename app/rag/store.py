"""向量库:负责文档块的持久化、索引重建与相似度检索。

存储结构(全部落在 data/index/ 下):
    index.json   文档与块的元数据 + 向量化后端的状态(如 IDF)
    vectors.npy  与块顺序一一对应的归一化向量矩阵

设计取舍:TF-IDF 的 IDF 依赖全量语料,新增或删除文档后需要重建索引;
知识库规模较小时重建成本可以忽略,换来的是实现简单、结果可复现。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import BaseEmbedder

INDEX_FILE = "index.json"
VECTOR_FILE = "vectors.npy"
SCHEMA_VERSION = 1


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source: str
    created_at: str
    char_count: int
    chunk_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    index: int
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchHit:
    """一条召回结果,带来源信息便于在答案里做引用。"""

    text: str
    score: float
    doc_id: str
    title: str
    chunk_index: int
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class VectorStore:
    def __init__(self, index_dir: Path, embedder: BaseEmbedder) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: list[ChunkRecord] = []
        self.vectors: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)

    # ---------------- 持久化 ----------------

    @property
    def _index_path(self) -> Path:
        return self.index_dir / INDEX_FILE

    @property
    def _vector_path(self) -> Path:
        return self.index_dir / VECTOR_FILE

    def load(self) -> "VectorStore":
        if not self._index_path.exists():
            return self
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self

        self.embedder.load_state(payload.get("embedder", {}))
        self.documents = {
            d["doc_id"]: DocumentRecord(**d) for d in payload.get("documents", [])
        }
        self.chunks = [ChunkRecord(**c) for c in payload.get("chunks", [])]
        if self._vector_path.exists():
            vectors = np.load(self._vector_path)
            if vectors.shape[0] == len(self.chunks):
                self.vectors = vectors.astype(np.float32)
            else:                                   # 索引与元数据不一致时自动重建
                self._rebuild()
        return self

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "embedder": self.embedder.state(),
            "documents": [d.to_dict() for d in self.documents.values()],
            "chunks": [c.to_dict() for c in self.chunks],
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

        vectors = np.asarray(self.vectors, dtype=np.float32)
        np.save(self._vector_path, vectors)

    # ---------------- 索引 ----------------

    def _rebuild(self) -> None:
        """重新统计 IDF 并重算所有块的向量。"""
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.vectors = (
            self.embedder.embed_documents(texts)
            if texts
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )

    def add_document(self, title: str, source: str, chunks: list[Any]) -> DocumentRecord:
        """新增文档。chunks 为 splitter.Chunk 列表。"""
        doc_id = uuid.uuid4().hex[:12]
        text_len = sum(len(c.text) for c in chunks)
        record = DocumentRecord(
            doc_id=doc_id,
            title=title or source or f"文档-{doc_id}",
            source=source,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            char_count=text_len,
            chunk_count=len(chunks),
        )
        for chunk in chunks:
            self.chunks.append(
                ChunkRecord(
                    chunk_id=uuid.uuid4().hex[:12],
                    doc_id=doc_id,
                    title=record.title,
                    text=chunk.text,
                    index=chunk.index,
                    start=chunk.start,
                    end=chunk.end,
                )
            )
        self.documents[doc_id] = record
        self._rebuild()
        self.save()
        return record

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._rebuild()
        self.save()
        return True

    # ---------------- 检索 ----------------

    def search(self, query: str, top_k: int = 4, threshold: float = 0.0) -> list[SearchHit]:
        if not query.strip() or not self.chunks or self.vectors.shape[0] == 0:
            return []
        query_vec = self.embedder.embed_query(query)
        if query_vec.shape[0] != self.vectors.shape[1]:
            return []
        scores = self.vectors @ query_vec                       # 已归一化,点积即余弦相似度
        order = np.argsort(-scores)[: max(1, top_k)]
        hits: list[SearchHit] = []
        for i in order:
            score = float(scores[int(i)])
            if score < threshold:
                continue
            chunk = self.chunks[int(i)]
            hits.append(
                SearchHit(
                    text=chunk.text,
                    score=round(score, 4),
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    chunk_index=chunk.index,
                    chunk_id=chunk.chunk_id,
                )
            )
        return hits

    # ---------------- 查询 ----------------

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "characters": sum(c.end - c.start for c in self.chunks),
        }