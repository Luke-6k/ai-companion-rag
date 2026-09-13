"""端到端冒烟测试:真实启动服务,依次验证知识库与对话接口。

不依赖外部 API:通过 MOCK_LLM=true 走离线演示模式,因此可以在 CI 或没网的环境运行。
用法:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("SMOKE_PORT", "8123"))
BASE = f"http://127.0.0.1:{PORT}"
DATA_DIR = ROOT / ".tmp" / f"smoke-{uuid.uuid4().hex[:8]}"

SAMPLE_DOC = """星桥知识助手使用说明

账号登录:新员工由部门管理员创建账号,初始密码发送到公司邮箱,首次登录必须修改密码。
知识库上传:支持 txt、markdown、pdf 三种格式,单个文件不超过 10MB。
索引重建:删除文档后会同步删除该文档的所有片段并重建索引。
"""


def call(method: str, path: str, payload: dict | None = None, timeout: int = 60, raw: bool = False):
    url = BASE + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return body if raw else json.loads(body)


def wait_for_server(proc: subprocess.Popen, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"服务进程提前退出,返回码 {proc.returncode}")
        try:
            call("GET", "/health", timeout=3)
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("等待服务启动超时")


def main() -> int:
    env = dict(os.environ)
    env.update({
        "MOCK_LLM": "true",
        "DATA_DIR": str(DATA_DIR),
        "PYTHONPATH": os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep),
        "PYTHONIOENCODING": "utf-8",
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )

    passed: list[str] = []
    try:
        wait_for_server(proc)
        print("[1/6] 服务已启动, /health 正常")

        index = call("GET", "/")
        assert "endpoints" in index, "接口索引异常"
        print(f"[2/6] 接口索引正常,共 {len(index['endpoints'])} 个端点")

        doc = call("POST", "/api/v1/kb/documents/text",
                   {"title": "使用说明", "text": SAMPLE_DOC})
        assert doc["chunk_count"] >= 1, "文档未切分出片段"
        print(f"[3/6] 文档入库成功: {doc['title']}, 切分 {doc['chunk_count']} 块")

        hits = call("POST", "/api/v1/kb/search", {"query": "支持哪些文件格式", "top_k": 3})
        assert hits, "检索没有召回结果"
        assert any("10MB" in h["text"] or "pdf" in h["text"] for h in hits), "召回内容不相关"
        print(f"[4/6] 检索命中 {len(hits)} 条, top1 相似度 {hits[0]['score']}")

        answer = call("POST", "/api/v1/chat", {"question": "单个文件大小限制是多少"})
        assert answer["answer"], "问答返回为空"
        assert answer["sources"], "问答未返回引用来源"
        assert answer["session_id"], "未返回会话 ID"
        print(f"[5/6] 问答成功, 引用 {len(answer['sources'])} 条来源, 会话 {answer['session_id']}")

        raw = call("POST", "/api/v1/chat/stream",
                   {"question": "删除文档会发生什么", "session_id": answer["session_id"]},
                   raw=True, timeout=60)
        events = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line[6:].strip() != "[DONE]"]
        kinds = [e["type"] for e in events]
        assert "sources" in kinds and "delta" in kinds and "done" in kinds, f"SSE 事件不完整: {kinds}"
        text = "".join(e["text"] for e in events if e["type"] == "delta")
        print(f"[6/6] 流式问答正常: {len(kinds)} 个事件, 回复 {len(text)} 字")

        sessions = call("GET", "/api/v1/sessions")
        assert sessions["total"] >= 1, "会话未落库"
        deleted = call("DELETE", f"/api/v1/kb/documents/{doc['doc_id']}")
        assert deleted["deleted"] is True, "删除文档失败"
        health = call("GET", "/health")
        assert health["knowledge_base"]["documents"] == 0, "删除后知识库统计未更新"
        passed.append("全部通过")

        print("\n=== 冒烟测试全部通过 ===")
        return 0
    except AssertionError as exc:
        print(f"\n[失败] 断言不通过: {exc}")
        return 1
    except Exception as exc:                       # noqa: BLE001
        print(f"\n[失败] {type(exc).__name__}: {exc}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdout:
            logs = proc.stdout.read()
            if logs.strip():
                print("\n--- 服务日志 ---")
                print(logs.strip()[:2000])


if __name__ == "__main__":
    sys.exit(main())