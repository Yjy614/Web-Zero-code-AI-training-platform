"""OpenAI 兼容 Chat Completions 客户端（支持 tools / tool_calls / 流式输出）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
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


class LLMStreamCancelled(RuntimeError):
    """流式生成被用户取消。"""


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class ChatMessageResult:
    """助手返回的一条 message。"""

    role: str
    content: str | None = None
    reasoning_content: str | None = None  # DeepSeek reasoner 等模型的思考过程
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ChatCompletionResult:
    """一次 chat/completions 的结构化结果。"""

    message: ChatMessageResult
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    thinking_enabled: bool | None,
    reasoning_effort: str | None,
    stream: bool,
) -> dict[str, Any]:
    if not model:
        raise ValueError("未配置大模型 model")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if stream:
        payload["stream"] = True
    # 思考模式下 temperature 等无效；关闭思考时再传 temperature
    use_thinking = thinking_enabled is True
    if use_thinking:
        payload["thinking"] = {"type": "enabled"}
        effort = (reasoning_effort or "high").strip().lower()
        if effort not in {"low", "high", "max"}:
            effort = "high"
        payload["reasoning_effort"] = effort
    elif thinking_enabled is False:
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = temperature
    else:
        # 未指定：不传 thinking，保持对非 DeepSeek 兼容接口友好
        payload["temperature"] = temperature

    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        else:
            payload["tool_choice"] = "auto"
    return payload


def _parse_message_dict(message: dict[str, Any]) -> ChatMessageResult:
    content = message.get("content")
    content_str = str(content).strip() if content is not None and str(content).strip() else None

    # DeepSeek reasoner / 部分兼容接口会返回 reasoning_content
    reasoning_raw = message.get("reasoning_content")
    if reasoning_raw is None:
        reasoning_raw = message.get("reasoning")
    reasoning_str = (
        str(reasoning_raw).strip() if reasoning_raw is not None and str(reasoning_raw).strip() else None
    )

    tool_calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            args = fn.get("arguments")
            if isinstance(args, dict):
                args_s = json.dumps(args, ensure_ascii=False)
            else:
                args_s = str(args or "{}")
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{len(tool_calls)}"),
                    name=name,
                    arguments=args_s,
                )
            )

    return ChatMessageResult(
        role=str(message.get("role") or "assistant"),
        content=content_str,
        reasoning_content=reasoning_str,
        tool_calls=tool_calls,
    )


def chat_completion_exchange(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int = 60,
    temperature: float = 0.4,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
) -> ChatCompletionResult:
    """
    调用兼容 OpenAI 的 chat/completions，返回结构化结果（可含 tool_calls）。
    messages / tools 均为 OpenAI 兼容格式。
    thinking_enabled / reasoning_effort 对应 DeepSeek 思考模式参数。
    """
    url = _chat_completions_url(base_url)
    payload = _build_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
        stream=False,
    )

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
            raw_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"大模型接口 HTTP {e.code}：{detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接大模型服务：{e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError("大模型请求超时，请增大设置中的 timeout 或稍后重试") from e

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError("大模型返回非 JSON") from e

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("大模型返回缺少 choices")
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("大模型返回缺少 message")

    parsed = _parse_message_dict(message)
    if not parsed.content and not parsed.tool_calls and not parsed.reasoning_content:
        raise RuntimeError("大模型返回空内容且无工具调用")

    finish = choice0.get("finish_reason")
    return ChatCompletionResult(
        message=parsed,
        finish_reason=str(finish) if finish is not None else None,
        raw=data if isinstance(data, dict) else {},
    )


def chat_completion_exchange_stream(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int = 60,
    temperature: float = 0.4,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
    on_delta: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ChatCompletionResult:
    """
    流式调用 chat/completions：边收边通过 on_delta 回调增量。
    on_delta 事件：
      {"kind": "reasoning", "delta": "..."}
      {"kind": "content", "delta": "..."}
    最终仍返回与非流式一致的 ChatCompletionResult（含合并后的 tool_calls）。
    """
    url = _chat_completions_url(base_url)
    payload = _build_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
        stream=True,
    )

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    # 流式：单次阻塞读超时；整体时长由取消检查与客户端断开控制
    read_timeout = max(30, int(timeout or 60))

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    # index -> 累加中的 tool call
    tool_acc: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    role = "assistant"
    raw_last: dict[str, Any] = {}

    def _emit_delta(kind: str, delta: str) -> None:
        if not delta or not on_delta:
            return
        on_delta({"kind": kind, "delta": delta})

    try:
        with urllib.request.urlopen(req, timeout=read_timeout) as resp:
            while True:
                if should_cancel and should_cancel():
                    raise LLMStreamCancelled("用户已终止本轮思考")
                line_b = resp.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    # SSE 注释/心跳
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    raw_last = chunk
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice0 = choices[0] if isinstance(choices[0], dict) else {}
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = str(fr)
                delta = choice0.get("delta") if isinstance(choice0, dict) else None
                if not isinstance(delta, dict):
                    continue
                if delta.get("role"):
                    role = str(delta.get("role") or role)

                # 思考增量（DeepSeek / 兼容接口）
                r_delta = delta.get("reasoning_content")
                if r_delta is None:
                    r_delta = delta.get("reasoning")
                if r_delta is not None:
                    piece = str(r_delta)
                    if piece:
                        reasoning_parts.append(piece)
                        _emit_delta("reasoning", piece)

                c_delta = delta.get("content")
                if c_delta is not None:
                    piece = str(c_delta)
                    if piece:
                        content_parts.append(piece)
                        _emit_delta("content", piece)

                # 工具调用增量合并
                raw_tcs = delta.get("tool_calls")
                if isinstance(raw_tcs, list):
                    for tc in raw_tcs:
                        if not isinstance(tc, dict):
                            continue
                        try:
                            idx = int(tc.get("index", 0))
                        except (TypeError, ValueError):
                            idx = 0
                        slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = str(tc["id"])
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        if fn.get("name"):
                            slot["name"] = str(fn["name"])
                        if fn.get("arguments") is not None:
                            slot["arguments"] += str(fn.get("arguments") or "")
    except LLMStreamCancelled:
        raise
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"大模型接口 HTTP {e.code}：{detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接大模型服务：{e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError("大模型请求超时，请增大设置中的 timeout 或稍后重试") from e

    if should_cancel and should_cancel():
        raise LLMStreamCancelled("用户已终止本轮思考")

    content_str = "".join(content_parts).strip() or None
    reasoning_str = "".join(reasoning_parts).strip() or None
    tool_calls: list[ToolCall] = []
    for idx in sorted(tool_acc.keys()):
        slot = tool_acc[idx]
        name = (slot.get("name") or "").strip()
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                id=(slot.get("id") or f"call_{idx}").strip() or f"call_{idx}",
                name=name,
                arguments=slot.get("arguments") or "{}",
            )
        )

    if not content_str and not tool_calls and not reasoning_str:
        raise RuntimeError("大模型返回空内容且无工具调用")

    return ChatCompletionResult(
        message=ChatMessageResult(
            role=role,
            content=content_str,
            reasoning_content=reasoning_str,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
        raw=raw_last,
    )


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int = 60,
    temperature: float = 0.4,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """
    调用兼容 OpenAI 的 chat/completions，返回助手文本。
    失败时抛出 RuntimeError（含中文说明）。
    兼容旧调用方；需要工具调用时请使用 chat_completion_exchange。
    """
    result = chat_completion_exchange(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        timeout=timeout,
        temperature=temperature,
        tools=None,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
    )
    content = result.message.content
    if not content:
        raise RuntimeError("大模型返回空内容")
    return content
