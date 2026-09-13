"""大模型客户端:统一封装 OpenAI 兼容协议(默认指向 DeepSeek)。

额外提供两种降级能力,便于本地开发与自动化测试:
    * MOCK_LLM=true  不调用真实接口,返回可预测的模拟回答(离线演示 / CI 用);
    * 统一的异常翻译,把底层报错转成用户能看懂的提示。
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .config import settings


class LLMError(RuntimeError):
    """模型调用失败,message 已是可直接展示给用户的文案。"""


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if any(k in low for k in ("api key", "authentication", "unauthorized", "401")):
        return "API Key 无效或未配置,请检查环境变量 DEEPSEEK_API_KEY。"
    if any(k in low for k in ("insufficient", "balance", "quota", "402")):
        return "账户余额不足或超出配额,请前往开放平台充值后再试。"
    if any(k in low for k in ("rate limit", "429", "too many requests")):
        return "请求过于频繁,已触发限流,请稍后重试。"
    if any(k in low for k in ("timeout", "timed out")):
        return "调用模型超时,请稍后重试。"
    if any(k in low for k in ("connection", "network", "ssl", "proxy", "getaddrinfo")):
        return "网络异常,无法连接模型服务,请检查网络或代理设置。"
    return f"调用模型失败: {text}"


class LLMClient:
    def __init__(self, api_key: str = "", base_url: str = "", model: str = "",
                 temperature: float | None = None, timeout: float = 120.0,
                 mock: bool = False) -> None:
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.timeout = timeout
        self.mock = mock or settings.mock_llm
        self._client = None

    @property
    def available(self) -> bool:
        return self.mock or bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise LLMError("未配置 DEEPSEEK_API_KEY,无法调用大模型(可设置 MOCK_LLM=true 进入离线演示模式)")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    @staticmethod
    def _mock_reply(messages: list[dict]) -> str:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        system = messages[0]["content"] if messages else ""
        if "参考资料" in system:
            return f"[离线演示模式] 已检索到知识库内容,针对问题「{question}」的答案将在这里生成 [1]。"
        return f"[离线演示模式] 收到你的消息:「{question}」,这里会返回模型的真实回复。"

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """流式返回模型增量文本。"""
        if self.mock:
            for piece in self._mock_reply(messages):
                yield piece
            return
        try:
            client = self._get_client()
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except LLMError:
            raise
        except Exception as exc:                     # noqa: BLE001 统一翻译后抛出
            raise LLMError(friendly_error(exc)) from exc

    def chat(self, messages: list[dict]) -> str:
        return "".join(self.stream_chat(messages))

    def health(self) -> dict:
        return {
            "ready": self.available,
            "mock": self.mock,
            "model": self.model,
            "base_url": self.base_url,
        }