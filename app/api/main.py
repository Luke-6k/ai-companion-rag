"""FastAPI 应用入口:知识库管理 + RAG 对话 + 会话管理。

接口一览:
    GET    /health                        健康检查(含模型与索引状态)
    GET    /api/v1/kb/documents           知识库文档列表
    POST   /api/v1/kb/documents           上传文档(txt / md / pdf)
    POST   /api/v1/kb/documents/text      直接提交文本入库
    DELETE /api/v1/kb/documents/{doc_id}  删除文档并重建索引
    POST   /api/v1/kb/search              检索调试(返回命中片段与相似度)
    POST   /api/v1/chat                   问答(非流式)
    POST   /api/v1/chat/stream            问答(SSE 流式,逐字返回)
    GET    /api/v1/sessions               会话列表
    GET    /api/v1/sessions/{id}          会话详情
    DELETE /api/v1/sessions/{id}          删除会话
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..config import settings
from ..llm import LLMClient
from ..rag.embedder import build_embedder
from ..rag.loader import UnsupportedFileType
from ..rag.store import VectorStore
from ..schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentItem,
    DocumentListResponse,
    SearchRequest,
    SourceItem,
    TextDocumentRequest,
)
from ..services.chat_service import ChatService
from ..services.kb_service import KBService
from ..services.session_service import SessionService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@lru_cache
def get_store() -> VectorStore:
    settings.ensure_dirs()
    store = VectorStore(settings.index_dir, build_embedder(
        backend=settings.embedding_backend,
        dim=settings.embedding_dim,
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    ))
    return store.load()


@lru_cache
def get_kb() -> KBService:
    return KBService(get_store(), settings.chunk_size, settings.chunk_overlap)


@lru_cache
def get_sessions() -> SessionService:
    settings.ensure_dirs()
    return SessionService(settings.session_dir)


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache
def get_chat() -> ChatService:
    return ChatService(
        kb=get_kb(),
        llm=get_llm(),
        sessions=get_sessions(),
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        history_rounds=settings.history_rounds,
    )


app = FastAPI(
    title="AI 伴侣 · RAG 知识库问答服务",
    description="Streamlit 前端 + FastAPI 后端 + 可插拔向量检索的 AI 应用示例",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="接口索引")
def index() -> dict:
    return {
        "service": "ai-companion-rag",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "GET /api/v1/kb/documents",
            "POST /api/v1/kb/documents",
            "POST /api/v1/kb/documents/text",
            "DELETE /api/v1/kb/documents/{doc_id}",
            "POST /api/v1/kb/search",
            "POST /api/v1/chat",
            "POST /api/v1/chat/stream",
            "GET /api/v1/sessions",
        ],
    }


@app.get("/health", summary="健康检查")
def health() -> dict:
    return {
        "status": "ok",
        "llm": get_llm().health(),
        "knowledge_base": get_kb().stats(),
    }


# ---------------------------------------------------------------- 知识库


@app.get("/api/v1/kb/documents", response_model=DocumentListResponse, summary="文档列表")
def list_documents(kb: KBService = Depends(get_kb)) -> DocumentListResponse:
    items = [DocumentItem(**doc.to_dict()) for doc in kb.list_documents()]
    return DocumentListResponse(total=len(items), items=items)


@app.post("/api/v1/kb/documents", response_model=DocumentItem, summary="上传文档")
async def upload_document(file: UploadFile = File(...), kb: KBService = Depends(get_kb)) -> DocumentItem:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过 10MB 限制")
    try:
        record = kb.add_file(file.filename or "未命名文档", data)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentItem(**record.to_dict())


@app.post("/api/v1/kb/documents/text", response_model=DocumentItem, summary="提交文本入库")
def add_text_document(payload: TextDocumentRequest, kb: KBService = Depends(get_kb)) -> DocumentItem:
    try:
        record = kb.add_text(payload.text, title=payload.title or "手动录入")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentItem(**record.to_dict())


@app.delete("/api/v1/kb/documents/{doc_id}", response_model=DeleteResponse, summary="删除文档")
def delete_document(doc_id: str, kb: KBService = Depends(get_kb)) -> DeleteResponse:
    if not kb.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return DeleteResponse(deleted=True, doc_id=doc_id)


@app.post("/api/v1/kb/search", response_model=list[SourceItem], summary="检索调试")
def search(payload: SearchRequest, kb: KBService = Depends(get_kb)) -> list[SourceItem]:
    hits = kb.search(payload.query, top_k=payload.top_k, threshold=payload.threshold)
    return [SourceItem(**hit.to_dict()) for hit in hits]


# ---------------------------------------------------------------- 对话


@app.post("/api/v1/chat", response_model=ChatResponse, summary="问答(非流式)")
def chat(payload: ChatRequest, service: ChatService = Depends(get_chat)) -> ChatResponse:
    result = service.answer(
        question=payload.question,
        session_id=payload.session_id,
        mode=payload.mode,
        top_k=payload.top_k,
        threshold=payload.threshold,
        nickname=payload.nickname or "小助手",
        character=payload.character or "",
    )
    if result["error"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        session_id=result["session_id"],
    )


@app.post("/api/v1/chat/stream", summary="问答(SSE 流式)")
def chat_stream(payload: ChatRequest, service: ChatService = Depends(get_chat)) -> StreamingResponse:
    def event_source() -> Iterator[str]:
        try:
            for event in service.stream_answer(
                question=payload.question,
                session_id=payload.session_id,
                mode=payload.mode,
                top_k=payload.top_k,
                threshold=payload.threshold,
                nickname=payload.nickname or "小助手",
                character=payload.character or "",
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:                       # noqa: BLE001 兜底,避免流中断无提示
            payload_error = {"type": "error", "message": f"服务异常: {exc}"}
            yield f"data: {json.dumps(payload_error, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- 会话


@app.get("/api/v1/sessions", summary="会话列表")
def list_sessions(sessions: SessionService = Depends(get_sessions)) -> dict:
    items = sessions.list()
    return {"total": len(items), "items": items}


@app.get("/api/v1/sessions/{session_id}", summary="会话详情")
def get_session(session_id: str, sessions: SessionService = Depends(get_sessions)) -> dict:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session.to_dict()


@app.delete("/api/v1/sessions/{session_id}", summary="删除会话")
def delete_session(session_id: str, sessions: SessionService = Depends(get_sessions)) -> dict:
    return {"deleted": sessions.delete(session_id), "session_id": session_id}