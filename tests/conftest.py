"""pytest 公共 fixture:为每个用例准备独立的临时目录。"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parent.parent / ".tmp"


@pytest.fixture
def work_dir():
    """提供一个隔离的临时目录,用例结束后自动清理。"""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / f"case-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)