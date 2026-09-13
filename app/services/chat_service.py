"""对话编排:把「检索 -> 提示词拼装 -> 大模型调用 -> 会话落库」串成一条链路。

两种模式:
    rag   知识库问答 —— 先检索相关片段,要求模型只依据资料作答并标注引用编号;
    chat  角色扮演闲聊 —— 不检索,按人设系统提示词对话。

对外暴露统一的流式事件协议,由 API 层转成 SSE:
    {"type": "sources", "sources": [...]}
    {"type": "delta",   "text": "..."}
    {"type": "done",    "session_id": "...", "answer": "...", "sources": [...]}
    {"type": "error",   "message": "..."}
"""

from __future__ import annotations

from typing import Iterator

from ..llm import LLMClient, LLMError
from ..rag.store import SearchHit
from .kb_service import KBService
from .session_service import Session, SessionService

RAG_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手,负责依据给定资料回答用户问题。

【回答要求】
1. 只依据下方【参考资料】作答,不要使用资料之外的信息,也不要编造;
2. 引用资料中的信息时,在句末标注来源编号,例如 [1]、[2];
3. 若参考资料不足以回答问题,直接说明"根据当前知识库无法回答该问题";
4. 使用与用户相同的语言,条理清晰,避免冗长。

【参考资料】
{context}
"""

PERSONA_SYSTEM_PROMPT = """你正在扮演用户的亲密伴侣「{nickname}」,请完全代入这个角色。

【扮演规则】
1. 每次只回复 1 条消息;
2. 只输出对话内容,不要写场景描述或旁白;
3. 使用与用户相同的语言,回复简短自然,像微信聊天一样;
4. 回复要体现伴侣的性格特征。

【伴侣性格】
{character}
"""


def format_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(无)"
    blocks = [f"[{i}] 来源:{hit.title}\n{hit.text}" for i, hit in enumerate(hits, 1)]
    return "\n\n".join(blocks)


class ChatService:
    def __init__(self, kb: KBService, llm: LLMClient, sessions: SessionService,
                 top_k: int = 4, score_threshold: float = 0.0, history_rounds: int = 10) -> None:
        self.kb = kb
        self.llm = llm
        self.sessions = sessions
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.history_rounds = history_rounds

    def build_messages(self, question: str, hits: list[SearchHit],
                       history: list[dict], mode: str = "rag",
                       nickname: str = "小助手", character: str = "") -> list[dict]:
        if mode == "rag":
            system = RAG_SYSTEM_PROMPT.format(context=format_context(hits))
        else:
            system = PERSONA_SYSTEM_PROMPT.format(nickname=nickname or "小助手",
                                                  character=character or "友好、耐心")
        messages = [{"role": "system", "content": system}]
        if self.history_rounds > 0:
            recent = [m for m in history if m.get("role") in ("user", "assistant")][-self.history_rounds * 2:]
            messages.extend({"role": m["role"], "content": m["content"]} for m in recent)
        messages.append({"role": "user", "content": question})
        return messages

    def stream_answer(self, question: str, session_id: str | None = None, mode: str = "rag",
                      top_k: int | None = None, threshold: float | None = None,
                      nickname: str = "小助手", character: str = "") -> Iterator[dict]:
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "message": "问题不能为空"}
            return

        session: Session = self.sessions.get_or_create(session_id)
        history = list(session.messages)

        hits: list[SearchHit] = []
        if mode == "rag":
            hits = self.kb.search(
                question,
                top_k=top_k or self.top_k,
                threshold=self.score_threshold if threshold is None else threshold,
            )

        messages = self.build_messages(question, hits, history, mode=mode,
                                       nickname=nickname, character=character)
        sources = [hit.to_dict() for hit in hits]
        yield {"type": "sources", "sources": sources, "session_id": session.session_id}

        self.sessions.append(session, "user", question)

        answer = ""
        try:
            for piece in self.llm.stream_chat(messages):
                answer += piece
                yield {"type": "delta", "text": piece}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc), "session_id": session.session_id}
            return

        self.sessions.append(session, "assistant", answer)
        yield {"type": "done", "session_id": session.session_id, "answer": answer, "sources": sources}

    def answer(self, question: str, session_id: str | None = None, mode: str = "rag",
               top_k: int | None = None, threshold: float | None = None,
               nickname: str = "小助手", character: str = "") -> dict:
        """非流式封装:内部仍走流式链路,便于复用错误处理与会话落库。"""
        sources: list[dict] = []
        text = ""
        session_id_out = session_id or ""
        error = ""
        for event in self.stream_answer(question, session_id, mode, top_k, threshold, nickname, character):
            if event["type"] == "sources":
                sources = event.get("sources", [])
                session_id_out = event.get("session_id", session_id_out)
            elif event["type"] == "delta":
                text += event["text"]
            elif event["type"] == "done":
                session_id_out = event.get("session_id", session_id_out)
            elif event["type"] == "error":
                error = event["message"]
        return {"answer": text, "sources": sources, "session_id": session_id_out, "error": error}