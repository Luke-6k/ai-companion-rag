"""集中式配置:全部通过环境变量注入,便于本地开发与容器化部署保持一致。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """运行期配置。字段名大写化后即环境变量名。"""

    # --- 大模型(OpenAI 兼容协议,默认指向 DeepSeek) ---
    llm_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY") or _env("LLM_API_KEY"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-v4-pro"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 1.0))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 120.0))
    mock_llm: bool = field(default_factory=lambda: _env_bool("MOCK_LLM", False))

    # --- 向量化后端:hash(本地,离线) 或 api(需配置 EMBEDDING_*) ---
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hash").lower())
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 4096))
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY"))
    embedding_base_url: str = field(
        default_factory=lambda: _env("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    )
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-m3"))

    # --- 检索参数 ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    score_threshold: float = field(default_factory=lambda: _env_float("SCORE_THRESHOLD", 0.10))
    history_rounds: int = field(default_factory=lambda: _env_int("HISTORY_ROUNDS", 10))

    # --- 存储路径 ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BASE_DIR / "data"))))

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.knowledge_dir, self.index_dir, self.session_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()