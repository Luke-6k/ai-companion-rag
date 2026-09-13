"""会话持久化:每个会话一个 JSON 文件,支持多轮上下文与历史回溯。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def new_session_id() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:4]


@dataclass
class Session:
    session_id: str
    title: str = "新会话"
    created_at: str = field(default_factory=now_str)
    updated_at: str = field(default_factory=now_str)
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SessionService:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
        return self.session_dir / f"{safe}.json"

    def create(self, session_id: str | None = None) -> Session:
        session = Session(session_id=session_id or new_session_id())
        self.save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Session(
            session_id=data.get("session_id", session_id),
            title=data.get("title", "新会话"),
            created_at=data.get("created_at", now_str()),
            updated_at=data.get("updated_at", now_str()),
            messages=data.get("messages", []) or [],
        )

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        return self.create(session_id)

    def save(self, session: Session) -> None:
        session.updated_at = now_str()
        if not session.title or session.title == "新会话":
            first_user = next((m["content"] for m in session.messages if m.get("role") == "user"), "")
            if first_user:
                text = " ".join(first_user.split())
                session.title = text[:16] + ("…" if len(text) > 16 else "")
        path = self._path(session.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def append(self, session: Session, role: str, content: str) -> Session:
        session.messages.append({"role": role, "content": content, "time": now_str()})
        self.save(session)
        return session

    def list(self) -> list[dict]:
        items: list[dict] = []
        for path in self.session_dir.glob("*.json"):
            session = self.get(path.stem)
            if session is None:
                continue
            items.append(
                {
                    "session_id": session.session_id,
                    "title": session.title,
                    "updated_at": session.updated_at,
                    "message_count": len(session.messages),
                }
            )
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_messages(self, session_id: str) -> Session:
        session = self.get_or_create(session_id)
        session.messages = []
        session.title = "新会话"
        self.save(session)
        return session