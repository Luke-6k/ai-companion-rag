"""检索评测脚本:用固定问题集量化 RAG 的召回质量。

流程:启动一个临时服务 -> 导入 data/knowledge 下的文档 -> 逐条提问做检索 ->
统计命中率与误召回率 -> 输出报告(同时写入 data/eval/report.md)。

评测指标:
    * 可回答问题 Top-1 / Top-3 命中率 —— 期望命中的片段是否被检索出来
    * 无答案问题拒答率 —— 知识库里没有的内容是否被正确过滤(不硬塞片段给模型)

用法:
    python scripts/eval_retrieval.py                 # 默认阈值取配置文件里的 SCORE_THRESHOLD
    python scripts/eval_retrieval.py --threshold 0.1 # 手动指定阈值,用于调参对比
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_CSV = ROOT / "data" / "eval" / "test_questions.csv"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
REPORT_PATH = ROOT / "data" / "eval" / "report.md"
PORT = int(os.environ.get("EVAL_PORT", "8155"))
BASE = f"http://127.0.0.1:{PORT}"


def call(method: str, path: str, payload: dict | None = None, timeout: int = 60):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_file(path: Path) -> dict:
    """用 multipart 上传文档(不依赖第三方 HTTP 库,手写表单体)。"""
    boundary = "----evalboundary" + uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode("utf-8"),
        b"Content-Type: application/octet-stream\r\n\r\n",
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        BASE + "/api/v1/kb/documents", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(proc: subprocess.Popen, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"服务提前退出,返回码 {proc.returncode}")
        try:
            call("GET", "/health", timeout=3)
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit("等待服务启动超时")


def load_questions() -> list[dict]:
    with QUESTIONS_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evaluate(threshold: float, top_k: int) -> int:
    questions = load_questions()
    data_dir = ROOT / ".tmp" / f"eval-{uuid.uuid4().hex[:8]}"
    env = dict(os.environ)
    env.update({
        "MOCK_LLM": "true",
        "DATA_DIR": str(data_dir),
        "PYTHONPATH": os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep),
        "PYTHONIOENCODING": "utf-8",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "error"],
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    rows: list[dict] = []
    try:
        wait_for_server(proc)
        for doc in sorted(KNOWLEDGE_DIR.glob("*.md")):
            record = upload_file(doc)
            print(f"  已导入 {record['title']} -> {record['chunk_count']} 个片段")
        stats = call("GET", "/health")["knowledge_base"]
        print(f"  知识库共 {stats['documents']} 篇文档 / {stats['chunks']} 个片段\n")

        for item in questions:
            question = item["question"]
            kind = item["type"]
            keyword = item["expected_keyword"]
            hits = call("POST", "/api/v1/kb/search",
                        {"query": question, "top_k": top_k, "threshold": threshold})

            rank, best_score, snippet = None, 0.0, ""
            if hits:
                best_score = hits[0]["score"]
                snippet = hits[0]["text"][:38].replace("\n", " ")
            if kind == "answerable":
                for i, hit in enumerate(hits, 1):
                    if keyword and keyword in hit["text"]:
                        rank = i
                        break
                ok = rank is not None
                result = "命中" if ok else "未命中"
            else:
                ok = not hits
                result = "正确拒答" if ok else "误召回"

            rows.append({"question": question, "kind": kind, "ok": ok, "rank": rank,
                         "score": best_score, "snippet": snippet, "result": result})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    answerable = [r for r in rows if r["kind"] == "answerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    top1 = sum(1 for r in answerable if r["rank"] == 1)
    top3 = sum(1 for r in answerable if r["rank"] is not None)
    refuse = sum(1 for r in unanswerable if r["ok"])

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a * 100.0 / b:.1f}%)" if b else "n/a"

    print("=" * 64)
    print(f"可回答问题 Top-1 命中率 : {pct(top1, len(answerable))}")
    print(f"可回答问题 Top-{top_k} 命中率 : {pct(top3, len(answerable))}")
    print(f"无答案问题正确拒答率   : {pct(refuse, len(unanswerable))}")
    print("=" * 64)

    # 控制台明细:只打印没达标的,避免刷屏
    missed = [r for r in rows if not r["ok"]]
    if missed:
        print("\n未达标的问题:")
        for r in missed:
            print(f"  [{r['result']}] {r['question']}  (最高分 {r['score']})")
    else:
        print("\n所有问题均达标。")

    write_report(rows, stats, threshold, top_k, top1, top3, refuse, answerable, unanswerable)
    print(f"\n报告已写入: {REPORT_PATH}")
    return 0 if not missed else 1


def write_report(rows, stats, threshold, top_k, top1, top3, refuse, answerable, unanswerable) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 检索评测报告",
        "",
        f"- 评测时间:{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 知识库规模:{stats['documents']} 篇文档 / {stats['chunks']} 个片段",
        f"- 检索参数:阈值 `{threshold}`,Top-K `{top_k}`,向量维度 `{stats['dim']}`({stats['embedder']})",
        f"- 问题集:{len(rows)} 条(可回答 {len(answerable)} 条 + 不可回答 {len(unanswerable)} 条)",
        "",
        "## 汇总",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| 可回答问题 Top-1 命中率 | {top1}/{len(answerable)} |",
        f"| 可回答问题 Top-{top_k} 命中率 | {top3}/{len(answerable)} |",
        f"| 无答案问题正确拒答率 | {refuse}/{len(unanswerable)} |",
        "",
        "## 明细",
        "",
        "| # | 类型 | 问题 | 结果 | 命中排名 | 最高相似度 | 召回片段开头 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        kind = "可回答" if r["kind"] == "answerable" else "无答案"
        rank = str(r["rank"]) if r["rank"] else "-"
        lines.append(
            f"| {i} | {kind} | {r['question']} | {r['result']} | {rank} | "
            f"{r['score']} | {r['snippet']} |"
        )
    lines += [
        "",
        "> 说明:命中排名指第几条召回结果里出现了预期关键词;"
        "无答案问题的「正确拒答」表示该问题在阈值过滤后没有任何片段被召回。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--threshold", type=float, default=None, help="相似度阈值,默认取配置文件的值")
    parser.add_argument("--top-k", type=int, default=3, help="评测时召回的片段数,默认 3")
    args = parser.parse_args()

    if args.threshold is None:
        from app.config import settings

        threshold = settings.score_threshold
    else:
        threshold = args.threshold

    print(f"开始评测:阈值 {threshold}, Top-K {args.top_k}")
    return evaluate(threshold, args.top_k)


if __name__ == "__main__":
    sys.exit(main())