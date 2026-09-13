"""文档加载:支持 txt / md / pdf,统一转成纯文本。"""

from __future__ import annotations

import io
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}


class UnsupportedFileType(Exception):
    pass


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def load_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:                      # pragma: no cover
        raise UnsupportedFileType("解析 PDF 需要安装 pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def load_bytes(filename: str, data: bytes) -> str:
    """按扩展名选择解析方式,返回纯文本。"""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf(data)
    if suffix in TEXT_SUFFIXES or not suffix:
        return _decode(data)
    raise UnsupportedFileType(f"暂不支持的文件类型: {suffix}（支持 txt / md / pdf）")


def load_path(path: Path) -> str:
    path = Path(path)
    return load_bytes(path.name, path.read_bytes())