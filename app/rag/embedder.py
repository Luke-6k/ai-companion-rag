"""向量化层:抽象出统一接口,支持「本地稀疏向量」与「API 稠密向量」两种后端。

默认后端(hash)完全离线可用,不依赖任何外部服务,便于本地开发与单元测试;
生产环境可以切换到真正的 embedding 模型(EMBEDDING_BACKEND=api)。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str, ngrams: tuple[int, ...] = (1, 2)) -> list[str]:
    """中英混排的轻量切词:英文按单词,中文按字,再补 n-gram。"""
    base = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for n in ngrams:
        if n == 1:
            tokens.extend(base)
        else:
            tokens.extend("".join(base[i:i + n]) for i in range(len(base) - n + 1))
    return tokens


def _bucket(token: str, dim: int) -> int:
    """稳定哈希(hashlib 不受 PYTHONHASHSEED 影响)。"""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class BaseEmbedder(ABC):
    """向量化后端接口:实现 fit / embed_documents / embed_query 即可。"""

    name = "base"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """根据语料统计必要参数(如 IDF)。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """把文档批量转成 (n, dim) 的归一化向量矩阵。"""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """把查询转成 (dim,) 的归一化向量。"""

    def state(self) -> dict:
        return {"name": self.name, "dim": self.dim}

    def load_state(self, state: dict) -> None:
        self.dim = int(state.get("dim", self.dim))


class HashingNgramEmbedder(BaseEmbedder):
    """本地稀疏向量:字符 n-gram + 子线性 TF + IDF + L2 归一化。

    用哈希技巧把不定长词表压到固定维度,因此无需词表文件即可持久化;
    中文不再分词也能召回,代价是不具备语义泛化能力(近义词召回弱)。
    """

    name = "hash-ngram-tfidf"

    def __init__(self, dim: int = 1024, ngrams: tuple[int, ...] = (1, 2)) -> None:
        super().__init__(dim=dim)
        self.ngrams = ngrams
        self.idf = np.ones(dim, dtype=np.float32)

    def _counts(self, text: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for token in _tokenize(text, self.ngrams):
            idx = _bucket(token, self.dim)
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def fit(self, texts: list[str]) -> None:
        n_docs = max(1, len(texts))
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for idx in self._counts(text):
                df[idx] += 1
        self.idf = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = self._counts(text)
        if not counts:
            return vec
        for idx, tf in counts.items():
            vec[idx] = (1.0 + math.log(tf)) * float(self.idf[idx])
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def state(self) -> dict:
        return {"name": self.name, "dim": self.dim, "ngrams": list(self.ngrams),
                "idf": self.idf.tolist()}

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.ngrams = tuple(state.get("ngrams", (1, 2)))
        idf = state.get("idf")
        self.idf = np.asarray(idf, dtype=np.float32) if idf else np.ones(self.dim, dtype=np.float32)


class ApiEmbedder(BaseEmbedder):
    """调用 OpenAI 兼容的 embeddings 接口(如 SiliconFlow 的 BAAI/bge-m3)。

    需要配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL。
    """

    name = "api-embedding"

    def __init__(self, dim: int = 1024, model: str = "BAAI/bge-m3",
                 api_key: str = "", base_url: str = "") -> None:
        super().__init__(dim=dim)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError("EMBEDDING_API_KEY 未配置,无法使用 API 向量后端")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def fit(self, texts: list[str]) -> None:
        return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        vectors = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def state(self) -> dict:
        return {"name": self.name, "dim": self.dim, "model": self.model}


def build_embedder(backend: str = "hash", dim: int = 4096, **kwargs) -> BaseEmbedder:
    """工厂函数:按配置创建向量化后端。"""
    if backend == "api":
        return ApiEmbedder(dim=dim, model=kwargs.get("model", "BAAI/bge-m3"),
                           api_key=kwargs.get("api_key", ""), base_url=kwargs.get("base_url", ""))
    return HashingNgramEmbedder(dim=dim)