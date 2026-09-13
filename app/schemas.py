"""API 请求/响应模型(由 FastAPI 自动生成 OpenAPI 文档)。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID,不传则新建")
    mode: str = Field(default="rag", pattern="^(rag|chat)$", description="rag=知识库问答, chat=角色闲聊")
    top_k: int | None = Field(default=None, ge=1, le=20, description="召回条数")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="相似度阈值")
    nickname: str | None = Field(default=None, max_length=32, description="chat 模式下的角色昵称")
    character: str | None = Field(default=None, max_length=200, description="chat 模式下的角色性格")


class SourceItem(BaseModel):
    text: str
    score: float
    doc_id: str
    title: str
    chunk_index: int
    chunk_id: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    session_id: str = ""
    error: str = ""


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class TextDocumentRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    text: str = Field(..., min_length=1, description="文档正文")


class DocumentItem(BaseModel):
    doc_id: str
    title: str
    source: str
    created_at: str
    char_count: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentItem] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id: str