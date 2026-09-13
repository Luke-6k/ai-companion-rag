"""知识库服务:负责「文档 -> 切分 -> 向量化 -> 入库」以及检索入口。"""

from __future__ import annotations

from ..rag.loader import UnsupportedFileType, load_bytes, load_path
from ..rag.splitter import split_text
from ..rag.store import DocumentRecord, SearchHit, VectorStore


class KBService:
    def __init__(self, store: VectorStore, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def add_text(self, text: str, title: str = "", source: str = "") -> DocumentRecord:
        if not text.strip():
            raise ValueError("文档内容为空")
        chunks = split_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("文档切分后没有有效内容")
        return self.store.add_document(title=title, source=source, chunks=chunks)

    def add_file(self, filename: str, data: bytes) -> DocumentRecord:
        try:
            text = load_bytes(filename, data)
        except UnsupportedFileType:
            raise
        return self.add_text(text, title=filename, source=filename)

    def add_path(self, path) -> DocumentRecord:
        text = load_path(path)
        return self.add_text(text, title=str(path.name), source=str(path))

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id)

    def search(self, query: str, top_k: int = 4, threshold: float = 0.0) -> list[SearchHit]:
        return self.store.search(query, top_k=top_k, threshold=threshold)

    def stats(self) -> dict:
        return self.store.stats()