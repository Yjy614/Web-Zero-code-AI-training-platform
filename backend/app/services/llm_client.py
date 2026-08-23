"""OpenAI 兼容 Chat Completions 客户端（供 AI 建议等调用）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def _chat_completions_url(base_url: str) -> str:
    """规范化为 .../v1/chat/completions。"""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("未配置大模型 base_url")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 60,
    temperature: float = 0.4,
) -> str:
    """
    调用兼容 OpenAI 的 chat/completions，返回助手文本。
    失败时抛出 RuntimeError（含中文说明）。
    """
    if not model:
        raise ValueError("未配置大模型 model")
    url = _chat_completions_url(base_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(5, int(timeout or 60))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"大模型接口 HTTP {e.code}：{detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接大模型服务：{e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError("大模型请求超时，请增大设置中的 timeout 或稍后重试") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("大模型返回非 JSON") from e

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("大模型返回缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not content or not str(content).strip():
        raise RuntimeError("大模型返回空内容")
    return str(content).strip()
