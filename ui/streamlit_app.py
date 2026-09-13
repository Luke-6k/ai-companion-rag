"""Streamlit 前端:只负责界面与交互,所有能力都通过 FastAPI 接口调用。

这样前后端可以分开部署与扩容,也方便用 curl / Postman 直接调试后端能力。
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60

st.set_page_config(page_title="AI 伴侣 · 知识库问答", page_icon="🪬", layout="wide")


# ------------------------------------------------------------------ 接口封装

def api_get(path: str):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"调用后端失败: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text)
            st.error(f"请求失败({resp.status_code}): {detail}")
            return None
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"调用后端失败: {exc}")
        return None


def api_delete(path: str):
    try:
        resp = requests.delete(f"{API_BASE}{path}", timeout=TIMEOUT)
        return resp.status_code < 400
    except requests.RequestException as exc:
        st.error(f"调用后端失败: {exc}")
        return False


def stream_chat(payload: dict) -> Iterator[dict]:
    """消费后端 SSE 事件流,逐个 yield 解析后的事件。"""
    try:
        with requests.post(f"{API_BASE}/api/v1/chat/stream", json=payload,
                           stream=True, timeout=TIMEOUT * 4) as resp:
            if resp.status_code >= 400:
                st.error(f"请求失败({resp.status_code})")
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                body = raw[6:].strip()
                if body == "[DONE]":
                    break
                try:
                    yield json.loads(body)
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as exc:
        yield {"type": "error", "message": f"连接后端失败: {exc}"}


# ------------------------------------------------------------------ 状态初始化

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "nickname" not in st.session_state:
    st.session_state.nickname = "永雏塔菲"
if "character" not in st.session_state:
    st.session_state.character = "喜欢撒娇、语气软糯可爱的塔菲"

st.title("AI 伴侣 · 知识库问答")
st.caption("Streamlit 前端 + FastAPI 后端 + 可插拔向量检索(RAG)")

health = api_get("/health") or {}
retrieval = health.get("retrieval", {}) if isinstance(health, dict) else {}
default_top_k = int(retrieval.get("top_k", 4))
default_threshold = float(retrieval.get("score_threshold", 0.10))

if health:
    llm = health.get("llm", {})
    kb = health.get("knowledge_base", {})
    if not llm.get("ready"):
        st.warning("后端未检测到 DEEPSEEK_API_KEY;可设置 MOCK_LLM=true 进入离线演示模式。")

# ------------------------------------------------------------------ 侧边栏

with st.sidebar:
    st.subheader("对话模式")
    mode_label = st.radio("模式", ["知识库问答(RAG)", "伴侣闲聊"], label_visibility="collapsed")
    mode = "rag" if mode_label.startswith("知识库") else "chat"

    if mode == "rag":
        top_k = st.slider("召回片段数", 1, 10, default_top_k)
        threshold = st.slider("相似度阈值", 0.0, 0.5, default_threshold, step=0.01,
                              help="低于该分数的片段会被丢弃,默认值取自后端配置(SCORE_THRESHOLD);"
                                   "调低召回更多但更容易答无关问题,调高更保守")
    else:
        st.text_input("昵称", key="nickname")
        st.text_area("性格", key="character", height=80)
        top_k, threshold = 4, 0.0

    st.divider()
    st.subheader("会话")
    if st.button("➕ 新建会话", width="stretch"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.session_id = None
        st.rerun()
    sessions = api_get("/api/v1/sessions") or {}
    for item in sessions.get("items", [])[:8]:
        label = ("🟢 " if item["session_id"] == st.session_state.session_id else "") + item["title"]
        if st.button(label, key=f"load_{item['session_id']}", width="stretch"):
            detail = api_get(f"/api/v1/sessions/{item['session_id']}") or {}
            st.session_state.session_id = item["session_id"]
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in detail.get("messages", [])
            ]
            st.session_state.sources = []
            st.rerun()

    st.divider()
    st.subheader("知识库")
    uploaded = st.file_uploader("上传文档(txt / md / pdf)", type=["txt", "md", "markdown", "pdf"])
    if uploaded is not None and st.button("导入该文档", width="stretch"):
        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/kb/documents",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=TIMEOUT * 3,
            )
            if resp.status_code < 400:
                st.success(f"已导入:{resp.json()['title']}")
                st.rerun()
            else:
                st.error(f"导入失败({resp.status_code}):{resp.text[:200]}")
        except requests.RequestException as exc:
            st.error(f"上传失败:{exc}")

    docs = api_get("/api/v1/kb/documents") or {}
    if docs.get("items"):
        st.caption(f"共 {docs['total']} 篇文档")
        for doc in docs["items"]:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(f"{doc['title']}({doc['chunk_count']} 段)")
            with col2:
                if st.button("🗑️", key=f"del_{doc['doc_id']}", help="删除该文档并重建索引"):
                    if api_delete(f"/api/v1/kb/documents/{doc['doc_id']}"):
                        st.toast("文档已删除")
                        st.rerun()
    else:
        st.caption("知识库为空,上传文档后即可基于它提问。")

# ------------------------------------------------------------------ 主区域

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🪬" if message["role"] == "assistant" else "🧑"):
        st.markdown(message["content"])

if st.session_state.sources:
    with st.expander(f"📚 本次回答引用的 {len(st.session_state.sources)} 条资料", expanded=False):
        for i, src in enumerate(st.session_state.sources, 1):
            st.markdown(f"**[{i}] {src['title']}** · 相似度 `{src['score']}`")
            st.caption(src["text"][:300] + ("…" if len(src["text"]) > 300 else ""))

prompt = st.chat_input("输入问题,我会先检索知识库再回答")
if prompt and prompt.strip():
    question = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "mode": mode,
        "top_k": top_k,
        "threshold": threshold,
        "nickname": st.session_state.nickname,
        "character": st.session_state.character,
    }

    collected: list[str] = []
    sources: list[dict] = []
    error = ""

    with st.chat_message("assistant", avatar="🪬"):
        placeholder = st.empty()
        for event in stream_chat(payload):
            kind = event.get("type")
            if kind == "sources":
                sources = event.get("sources", [])
                st.session_state.session_id = event.get("session_id") or st.session_state.session_id
            elif kind == "delta":
                collected.append(event.get("text", ""))
                placeholder.markdown("".join(collected) + " ▌")
            elif kind == "error":
                error = event.get("message", "未知错误")
        if error:
            placeholder.error(error)
        else:
            placeholder.markdown("".join(collected))
            st.session_state.messages.append({"role": "assistant", "content": "".join(collected)})
            st.session_state.sources = sources
            st.rerun()