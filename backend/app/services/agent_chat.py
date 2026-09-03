"""对话式 Agent：会话持久化 + 多轮 tool calling 循环。"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import storage_root_path
from app.core.task_types import TASK_TYPE_LABELS, normalize_task_type
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.agent import (
    AgentChatMessageOut,
    AgentChatSessionOut,
    AgentChatSessionSummary,
    AgentChatTurnOut,
)
from app.services import agent_tools, agent_train_ops, dataset_prelabel
from app.services.llm_client import LLMStreamCancelled, chat_completion_exchange_stream
from app.services.runtime_settings import get_runtime_settings

MAX_TOOL_ROUNDS = 8
# 发给大模型的历史消息上限（含 tool），过长时从尾部截取并保证 tool 成对
MAX_LLM_HISTORY_MESSAGES = 48

# 开训确认话术（方案 A：每次训练必须确认）
_TRAIN_CONFIRM_OPTIONS = ("确认开训", "我想改参数")
_TRAIN_CONFIRM_YES = {
    "确认开训",
    "确认",
    "开训",
    "开始训练",
    "同意开训",
    "可以开训",
    "按这个开训",
    "按此开训",
    "好的开训",
    "好，开训",
    "好的，开训",
}
_TRAIN_CONFIRM_CHANGE = {
    "我想改参数",
    "改参数",
    "修改参数",
    "先别训",
    "不要开训",
    "取消开训",
    "再改改",
}

# 标注复查后：是否进入训练准备
_INSPECT_PREP_OPTIONS_OK = ("开始准备训练", "先不训练，继续标注")
_INSPECT_PREP_OPTIONS_BLOCKED = ("继续标注", "先去上传图片")
_INSPECT_PREP_YES = {
    "开始准备训练",
    "准备训练",
    "进入训练准备",
    "可以准备训练",
    "去训练",
    "开始训练准备",
}
_INSPECT_PREP_NO = {
    "先不训练，继续标注",
    "继续标注",
    "先不训练",
    "再标一会儿",
    "先去上传图片",
    "去上传",
}

# 训后评估确认
_EVAL_CONFIRM_OPTIONS = ("确认评估", "先不评估")
_EVAL_CONFIRM_YES = {
    "确认评估",
    "开始评估",
    "评估",
    "可以评估",
    "去做评估",
}
_EVAL_CONFIRM_NO = {
    "先不评估",
    "不评估",
    "跳过评估",
    "先看看再说",
}

# 创建数据集确认（名称/任务类型由用户拍板）
_CREATE_DS_CONFIRM_OPTIONS = ("确认创建", "我想改一下")
_CREATE_DS_CONFIRM_YES = {
    "确认创建",
    "确认",
    "创建",
    "可以创建",
    "同意创建",
    "按这个创建",
    "就这个",
    "好的创建",
    "好，创建",
}
_CREATE_DS_CONFIRM_CHANGE = {
    "我想改一下",
    "我想改",
    "改一下",
    "修改",
    "先别创建",
    "不要创建",
    "取消创建",
    "换个名字",
    "换类型",
}

# 通用动作确认：凡会改状态 / 开 Job / 打开标注等，均需用户明确同意
_ACTION_CONFIRM_OPTIONS = ("确认执行", "先不执行")
_ACTION_CONFIRM_YES = {
    "确认执行",
    "确认",
    "执行",
    "可以",
    "同意",
    "好的",
    "好",
    "开始",
    "做吧",
    "可以执行",
    "按这个做",
}
_ACTION_CONFIRM_NO = {
    "先不执行",
    "不执行",
    "先不做",
    "不要",
    "取消",
    "算了",
    "先别",
}
# 只读查询：可直接执行，无需确认
_READONLY_TOOLS = frozenset(
    {
        "list_datasets",
        "inspect_dataset",
        "list_weights",
        "list_devices",
        "get_job_status",
        "list_models",
        "ask_user",
    }
)
# 专用确认闸（各自有独立话术）；其余写操作走通用确认
_SPECIALIZED_CONFIRM_TOOLS = frozenset(
    {
        "create_dataset",
        "start_train",
        "start_eval",
    }
)
# 通用确认覆盖的写操作（训练准备 ensure/set 由「开始准备训练」覆盖，开训另有专用闸）
_GENERIC_CONFIRM_TOOLS = frozenset(
    {
        "guide_dataset_upload",
        "open_annotate",
        "start_prelabel",
        "run_sample_infer",
        "export_model_onnx",
        "cancel_job",
    }
)

SYSTEM_PROMPT = """你是工业零代码 AI 训练平台的智能助手（对话式 Agent）。
你可以通过工具完成：查询/创建数据集、引导上传与标注、预标注、查看权重与设备、准备训练、设参开训、查 Job、评估、抽样推理、列模型库、导出 ONNX、取消任务；不确定时用 ask_user。

总原则：【你不能替用户决定做不做】。任何会改数据、创建资源、打开标注、启动 Job、导出、取消的动作，都必须先征得用户明确同意；系统也会强制二次确认，用户点确认前不会真正执行。查询类（列表、检查、进度）可以直接调用。

规则：
1. 先思考再行动：先理解用户意图；寒暄、概念解释、一般建议可直接回复，不要为了用工具而用工具。
2. 需要事实（数据集列表、标注统计、权重、Job 状态等）时再调用只读工具，不要编造。
3. 若准备做写操作：先用一两句中文说明「建议做什么、为什么」，再用工具；系统会弹出确认，在用户确认前不要假定已完成。
4. 用简洁中文回复；可用 Markdown。思考过程（reasoning_content）必须使用中文。
5. 创建数据集：名称与任务类型（detect/segment/pose）由用户决定，不要默认选型；已有集优先复用。
6. 创建完成后，告诉用户点对话框左下角 + 或「在对话中上传图片」；不要说聊天不能传文件，也不要优先跳转向导。
7. 标注只能由用户完成。open_annotate 前说清要标什么；用户说「我标完了」时立刻 inspect_dataset 复查。
8. 推荐流程：inspect_dataset →（用户确认进入训练准备）→ list_weights/list_devices（可直接查）→ ensure_train_task / set_train_config（用户已确认准备后可执行）→ start_train（系统强制确认开训）。
9. 训练 Job 结束后先总结，再问是否评估；不要擅自 start_eval。
10. 工具返回 wait_for_job=true 时停止继续调工具，告知已开始并等待进度。
11. 工具结果用通俗语言总结；有 actions 时提醒点下方按钮（上传优先对话内）。
"""

# 会话级取消标记（思考/工具循环）
_cancel_flags: dict[str, bool] = {}
_cancel_lock = threading.Lock()


class ChatTurnCancelled(Exception):
    """用户主动终止本轮对话思考/工具调用。"""


def request_chat_cancel(session_id: str) -> None:
    """请求终止指定会话当前一轮。"""
    sid = (session_id or "").strip()
    if not sid:
        return
    with _cancel_lock:
        _cancel_flags[sid] = True


def clear_chat_cancel(session_id: str) -> None:
    with _cancel_lock:
        _cancel_flags.pop(session_id, None)


def is_chat_cancelled(session_id: str) -> bool:
    with _cancel_lock:
        return bool(_cancel_flags.get(session_id))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chats_dir(owner_id: int) -> Path:
    d = storage_root_path() / "agent_chats" / f"user_{owner_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(owner_id: int, session_id: str) -> Path:
    return _chats_dir(owner_id) / f"{session_id}.json"


def save_session(session: AgentChatSessionOut) -> None:
    session.updated_at = _now_iso()
    path = _session_path(session.owner_id, session.id)
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")


def load_session(owner_id: int, session_id: str) -> AgentChatSessionOut | None:
    path = _session_path(owner_id, session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentChatSessionOut.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def list_sessions(owner_id: int, *, limit: int = 40) -> list[AgentChatSessionSummary]:
    root = _chats_dir(owner_id)
    items: list[AgentChatSessionSummary] = []
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[: max(1, limit)]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sess = AgentChatSessionOut.model_validate(data)
            items.append(
                AgentChatSessionSummary(
                    id=sess.id,
                    title=sess.title,
                    updated_at=sess.updated_at,
                    message_count=len([m for m in sess.messages if m.role in {"user", "assistant"}]),
                    status=sess.status,
                )
            )
        except (json.JSONDecodeError, ValueError):
            continue
    return items


def create_session(user: User, *, title: str | None = None) -> AgentChatSessionOut:
    sid = uuid.uuid4().hex[:12]
    now = _now_iso()
    session = AgentChatSessionOut(
        id=sid,
        owner_id=user.id,
        title=(title or "新对话").strip() or "新对话",
        status="idle",
        created_at=now,
        updated_at=now,
        messages=[],
        context={},
    )
    save_session(session)
    return session


def delete_session(owner_id: int, session_id: str) -> bool:
    path = _session_path(owner_id, session_id)
    if path.is_file():
        path.unlink()
        return True
    return False


def rename_session(owner_id: int, session_id: str, title: str) -> AgentChatSessionOut:
    """重命名会话标题。"""
    session = load_session(owner_id, session_id)
    if not session:
        raise ValueError("会话不存在")
    name = (title or "").strip()
    if not name:
        raise ValueError("标题不能为空")
    if len(name) > 80:
        name = name[:80]
    session.title = name
    save_session(session)
    return session


def _llm_settings() -> dict[str, Any]:
    """读取 Agent 用 LLM 配置（含思考模式）。"""
    settings = get_runtime_settings()
    llm = settings.get("llm") or {}
    base_url = str(llm.get("base_url") or "").strip()
    api_key = str(llm.get("api_key") or "").strip()
    model = str(llm.get("model") or "").strip()
    timeout = int(llm.get("timeout") or 120)
    if not (base_url and model):
        raise ValueError("未配置大模型：请在系统设置中填写 LLM 的 base_url 与 model")
    thinking_enabled = bool(llm.get("thinking_enabled", True))
    effort = str(llm.get("reasoning_effort") or "high").strip().lower()
    if effort not in {"low", "high", "max"}:
        effort = "high"
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "timeout": timeout,
        "thinking_enabled": thinking_enabled,
        "reasoning_effort": effort,
    }


def _msg(
    role: str,
    content: str | None = None,
    *,
    tool_call_id: str | None = None,
    name: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    ui: dict[str, Any] | None = None,
) -> AgentChatMessageOut:
    return AgentChatMessageOut(
        id=uuid.uuid4().hex[:10],
        role=role,  # type: ignore[arg-type]
        content=content,
        tool_call_id=tool_call_id,
        name=name,
        tool_calls=tool_calls or [],
        ui=ui or {},
        created_at=_now_iso(),
    )


def _trim_session_messages_for_llm(
    messages: list[AgentChatMessageOut],
) -> list[AgentChatMessageOut]:
    """截取近期历史，并向前扩展以保证 tool 消息与对应 assistant.tool_calls 成对。"""
    if len(messages) <= MAX_LLM_HISTORY_MESSAGES:
        return messages
    start = len(messages) - MAX_LLM_HISTORY_MESSAGES
    while start > 0 and messages[start].role == "tool":
        start -= 1
    # 若起点是带 tool_calls 的 assistant，保留它；若仍落在 tool，再退一格
    while start > 0 and messages[start].role == "tool":
        start -= 1
    return messages[start:]


def _skipped_tool_exec_result(tool_name: str, *, reason: str) -> dict[str, Any]:
    """为未执行的 tool_call 生成占位结果，保证对话历史成对。"""
    label = agent_tools.TOOL_LABELS.get(tool_name, tool_name)
    return {
        "ok": True,
        "data": {
            "skipped": True,
            "wait_for_user": False,
            "message": reason,
            "label": label,
        },
    }


def _append_skipped_tool_messages(
    session: AgentChatSessionOut,
    tool_calls: list[Any],
    *,
    reason: str,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    """为尚未回复的 tool_calls 补写 tool 消息。"""
    for tc in tool_calls:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        # openai 风格 dict: {"id", "function": {"name", "arguments"}}
        if name is None and isinstance(tc, dict):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name")
        tid = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
        name = str(name or "unknown")
        tid = str(tid or "")
        if not tid:
            continue
        exec_result = _skipped_tool_exec_result(name, reason=reason)
        label = agent_tools.TOOL_LABELS.get(name, name)
        if tool_events is not None:
            tool_events.append(
                {
                    "tool": name,
                    "label": label,
                    "ok": True,
                    "summary": "已跳过（等待用户确认）",
                }
            )
        session.messages.append(
            _msg(
                "tool",
                agent_tools.tool_content_for_llm(exec_result),
                tool_call_id=tid,
                name=name,
                ui={
                    "kind": "tool_result",
                    "tool": name,
                    "label": label,
                    "ok": True,
                    "summary": "已跳过（等待用户确认）",
                    "hidden": True,
                    "skipped": True,
                },
            )
        )


def _repair_openai_history_tool_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    修复发给大模型的历史：
    - assistant.tool_calls 后必须紧跟对应 tool 回复（缺则补占位）
    - 丢弃孤儿 tool（前面没有匹配的 tool_calls），避免 HTTP 400
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(items)
    while i < n:
        msg = items[i]
        role = msg.get("role")

        # 顶层孤儿 tool：直接丢弃（常见于 UI 注入的假 tool 气泡）
        if role == "tool":
            i += 1
            continue

        tool_calls = msg.get("tool_calls") if role == "assistant" else None
        if role != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
            out.append(msg)
            i += 1
            continue

        out.append(msg)
        i += 1
        expected: dict[str, str] = {}
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tid = str(tc.get("id") or "")
            if not tid:
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            expected[tid] = str(fn.get("name") or "unknown")

        got: set[str] = set()
        while i < n and items[i].get("role") == "tool":
            tmsg = items[i]
            tid = str(tmsg.get("tool_call_id") or "")
            if tid in expected and tid not in got:
                out.append(tmsg)
                got.add(tid)
            # 不匹配 / 重复的 tool 一律丢弃
            i += 1

        for tid, name in expected.items():
            if tid in got:
                continue
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": json.dumps(
                        {
                            "ok": True,
                            "data": {
                                "skipped": True,
                                "message": "该工具调用因等待用户确认或其他中断未执行。",
                                "tool": name,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    return out


def _to_openai_messages(
    session: AgentChatSessionOut,
    *,
    echo_reasoning: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = session.context or {}
    if ctx:
        # 上下文不要带预览图
        safe_ctx = {k: v for k, v in ctx.items() if k not in {"preview_base64", "image_base64"}}
        out.append(
            {
                "role": "system",
                "content": "当前会话上下文（JSON）：" + json.dumps(safe_ctx, ensure_ascii=False),
            }
        )
    history = _trim_session_messages_for_llm(list(session.messages or []))
    if len(session.messages or []) > len(history):
        out.append(
            {
                "role": "system",
                "content": "（提示：更早的对话已省略，请主要依据近期消息与当前上下文继续。）",
            }
        )
    for m in history:
        if m.role == "user":
            out.append({"role": "user", "content": m.content or ""})
        elif m.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            # 带 tools 的思考模式：必须回传历史 reasoning_content（官方要求）
            if echo_reasoning:
                reasoning = ""
                if isinstance(m.ui, dict):
                    reasoning = str(m.ui.get("reasoning") or "")
                item["reasoning_content"] = reasoning
            elif isinstance(m.ui, dict):
                reasoning = str(m.ui.get("reasoning") or "").strip()
                if reasoning:
                    item["reasoning_content"] = reasoning
            if m.tool_calls:
                item["tool_calls"] = m.tool_calls
                if not m.content:
                    item["content"] = None
            out.append(item)
        elif m.role == "tool":
            # 无 tool_call_id 或纯 UI 占位，不发给大模型
            tid = (m.tool_call_id or "").strip()
            if not tid:
                continue
            if isinstance(m.ui, dict) and (m.ui.get("hidden") or m.ui.get("skipped")):
                # skipped 占位仍需发给模型以成对；hidden 且非 skipped 才跳过
                if m.ui.get("skipped"):
                    pass
                else:
                    continue
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": m.content or "",
                }
            )
    return _repair_openai_history_tool_pairs(out)


def _maybe_update_title(session: AgentChatSessionOut, user_text: str) -> None:
    if session.title and session.title != "新对话":
        return
    title = user_text.strip().replace("\n", " ")
    if len(title) > 24:
        title = title[:24] + "…"
    session.title = title or "新对话"


def _apply_tool_side_effects(
    session: AgentChatSessionOut,
    tool_name: str,
    exec_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """根据工具结果更新 context / pending；返回 (pending_ask, pending_job, stop_loop)。"""
    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    pending_ask = None
    pending_job = None
    stop = False

    if not exec_result.get("ok"):
        return None, None, False

    if tool_name == "ask_user" and data.get("wait_for_user"):
        pending_ask = {"question": data.get("question"), "options": data.get("options") or []}
        stop = True

    # start_train 被硬闸拦截：转为等待用户确认
    if tool_name == "start_train" and data.get("need_confirm") and data.get("wait_for_user"):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or list(_TRAIN_CONFIRM_OPTIONS),
            "kind": "train_confirm",
        }
        if data.get("task_id") is not None:
            session.context = {
                **(session.context or {}),
                "pending_train_confirm_task_id": int(data["task_id"]),
                "task_id": data.get("task_id") or (session.context or {}).get("task_id"),
            }
        stop = True

    # inspect 后硬停：标注报告 + 是否进入训练准备
    if tool_name == "inspect_dataset" and data.get("need_confirm") and data.get("wait_for_user"):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or [],
            "kind": "inspect_prep",
        }
        if data.get("dataset_id") is not None:
            session.context = {
                **(session.context or {}),
                "pending_inspect_prep_dataset_id": int(data["dataset_id"]),
                "dataset_id": data.get("dataset_id"),
                "dataset_name": data.get("name") or data.get("dataset_name"),
                "task_type": data.get("task_type"),
                "can_train": data.get("can_train"),
                "labeled_count": data.get("labeled_count"),
            }
        stop = True

    # ensure/set 因未确认训练准备而被拦截（不含通用动作确认）
    if (
        tool_name in {"ensure_train_task", "set_train_config"}
        and data.get("need_confirm")
        and data.get("wait_for_user")
        and data.get("kind") != "action_confirm"
    ):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or list(_INSPECT_PREP_OPTIONS_OK),
            "kind": "inspect_prep",
        }
        if data.get("dataset_id") is not None:
            session.context = {
                **(session.context or {}),
                "pending_inspect_prep_dataset_id": int(data["dataset_id"]),
                "dataset_id": data.get("dataset_id"),
                "dataset_name": data.get("name") or (session.context or {}).get("dataset_name"),
            }
        stop = True

    # start_eval 硬闸
    if tool_name == "start_eval" and data.get("need_confirm") and data.get("wait_for_user"):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or list(_EVAL_CONFIRM_OPTIONS),
            "kind": "eval_confirm",
        }
        if data.get("task_id") is not None:
            session.context = {
                **(session.context or {}),
                "pending_eval_confirm_task_id": int(data["task_id"]),
                "task_id": data.get("task_id") or (session.context or {}).get("task_id"),
            }
        stop = True

    # create_dataset 硬闸：名称/任务类型需用户确认
    if tool_name == "create_dataset" and data.get("need_confirm") and data.get("wait_for_user"):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or list(_CREATE_DS_CONFIRM_OPTIONS),
            "kind": "create_dataset_confirm",
        }
        session.context = {
            **(session.context or {}),
            "pending_create_dataset_name": data.get("name"),
            "pending_create_dataset_task_type": data.get("task_type") or "detect",
        }
        stop = True

    # 通用动作硬闸
    if (
        tool_name in _GENERIC_CONFIRM_TOOLS
        and data.get("need_confirm")
        and data.get("wait_for_user")
        and data.get("kind") == "action_confirm"
    ):
        pending_ask = {
            "question": data.get("question"),
            "options": data.get("options") or list(_ACTION_CONFIRM_OPTIONS),
            "kind": "action_confirm",
        }
        stop = True

    if tool_name == "ensure_train_task":
        session.context = {
            **(session.context or {}),
            "task_id": data.get("task_id"),
            "task_name": data.get("task_name"),
            "dataset_id": data.get("dataset_id"),
            "dataset_name": data.get("dataset_name"),
            "task_type": data.get("task_type"),
            # 新任务/复用任务后需重新确认开训
            "train_start_approved_task_id": None,
        }

    if tool_name in {"create_dataset", "guide_dataset_upload", "open_annotate", "inspect_dataset"}:
        if data.get("dataset_id"):
            session.context = {
                **(session.context or {}),
                "dataset_id": data.get("dataset_id"),
                "dataset_name": data.get("name") or data.get("dataset_name"),
                "task_type": data.get("task_type"),
                "can_train": data.get("can_train", (session.context or {}).get("can_train")),
                "labeled_count": data.get("labeled_count", (session.context or {}).get("labeled_count")),
            }

    if tool_name == "set_train_config":
        session.context = {
            **(session.context or {}),
            "task_id": data.get("task_id"),
            "train_config": data.get("config"),
            "task_type": data.get("task_type") or (session.context or {}).get("task_type"),
            # 改参后必须重新确认
            "train_start_approved_task_id": None,
            "pending_train_confirm_task_id": None,
        }

    if tool_name in {"start_train", "start_eval", "start_prelabel", "export_model_onnx"} and data.get(
        "wait_for_job"
    ):
        pending_job = {
            "job_id": data.get("job_id"),
            "job_type": data.get("job_type") or tool_name.replace("start_", "").replace("export_model_", ""),
            "task_id": data.get("task_id"),
        }
        session.context = {
            **(session.context or {}),
            "task_id": data.get("task_id") or (session.context or {}).get("task_id"),
            "active_job_id": data.get("job_id"),
            "active_job_type": pending_job["job_type"],
        }
        stop = True

    if tool_name == "run_sample_infer":
        session.context = {
            **(session.context or {}),
            "last_infer": {
                "detection_count": data.get("detection_count"),
                "class_counts": data.get("class_counts"),
                "usable_hint": data.get("usable_hint"),
                "sample_image": data.get("sample_image"),
                "model_id": data.get("model_id"),
            },
        }

    return pending_ask, pending_job, stop


def _normalize_confirm_text(text: str) -> str:
    return (text or "").strip().replace(" ", "").replace("　", "")


def _is_train_confirm_yes(text: str) -> bool:
    t = _normalize_confirm_text(text)
    if t in {_normalize_confirm_text(x) for x in _TRAIN_CONFIRM_YES}:
        return True
    return ("确认" in t and "开训" in t) or t == "开始训练"


def _is_train_confirm_change(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _TRAIN_CONFIRM_CHANGE} or (
        "改" in t and "参数" in t
    )


def _is_inspect_prep_yes(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _INSPECT_PREP_YES} or (
        "准备" in t and "训练" in t
    )


def _is_inspect_prep_no(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _INSPECT_PREP_NO} or (
        ("继续" in t and "标注" in t) or ("上传" in t)
    )


def _is_eval_confirm_yes(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _EVAL_CONFIRM_YES} or (
        "确认" in t and "评估" in t
    )


def _is_eval_confirm_no(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _EVAL_CONFIRM_NO} or (
        "不" in t and "评估" in t
    )


def _is_create_dataset_confirm_yes(text: str) -> bool:
    t = _normalize_confirm_text(text)
    if t in {_normalize_confirm_text(x) for x in _CREATE_DS_CONFIRM_YES}:
        return True
    return ("确认" in t and "创建" in t) or t == "创建"


def _is_create_dataset_confirm_change(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _CREATE_DS_CONFIRM_CHANGE} or (
        ("改" in t) or ("换" in t) or ("取消" in t and "创建" in t) or ("不要" in t and "创建" in t)
    )


def _create_dataset_confirm_question(name: str, task_type: str) -> str:
    tt = normalize_task_type(task_type or "detect")
    label = TASK_TYPE_LABELS.get(tt, tt)
    return (
        "准备新建数据集，请你确认：\n"
        f"- 名称：{name}\n"
        f"- 类型：{label}\n"
        "没问题就点「确认创建」；要改名字或类型选「我想改一下」。"
    )


def _gate_create_dataset_result(
    session: AgentChatSessionOut,
    arguments: str,
) -> dict[str, Any] | None:
    """
    未确认前拦截 create_dataset；已确认则返回 None 继续真实创建。
    """
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    name = str(args.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "create_dataset 需要 name", "data": {}}
    tt = normalize_task_type(str(args.get("task_type") or "detect"))

    approved = (session.context or {}).get("create_dataset_approved")
    if isinstance(approved, dict):
        ap_name = str(approved.get("name") or "").strip()
        ap_tt = normalize_task_type(str(approved.get("task_type") or "detect"))
        if ap_name == name and ap_tt == tt:
            return None

    session.context = {
        **(session.context or {}),
        "pending_create_dataset_name": name,
        "pending_create_dataset_task_type": tt,
    }
    return {
        "ok": True,
        "data": {
            "need_confirm": True,
            "wait_for_user": True,
            "question": _create_dataset_confirm_question(name, tt),
            "options": list(_CREATE_DS_CONFIRM_OPTIONS),
            "name": name,
            "task_type": tt,
            "task_type_label": TASK_TYPE_LABELS.get(tt, tt),
            "message": "尚未获得用户确认，未创建数据集。",
        },
    }


def _consume_create_dataset_confirm_reply(
    session: AgentChatSessionOut, user_text: str
) -> dict[str, str] | None | str:
    """
    处理创建数据集确认。
    返回 {'name','task_type'} 表示确认创建；
    返回 'change' 表示用户要改；
    返回 None 表示未命中确认话术。
    """
    ctx = dict(session.context or {})
    pending_name = str(ctx.get("pending_create_dataset_name") or "").strip()
    if not pending_name:
        return None
    pending_tt = normalize_task_type(str(ctx.get("pending_create_dataset_task_type") or "detect"))

    if _is_create_dataset_confirm_change(user_text):
        ctx.pop("pending_create_dataset_name", None)
        ctx.pop("pending_create_dataset_task_type", None)
        ctx.pop("create_dataset_approved", None)
        session.context = ctx
        return "change"

    if _is_create_dataset_confirm_yes(user_text):
        ctx["create_dataset_approved"] = {"name": pending_name, "task_type": pending_tt}
        ctx.pop("pending_create_dataset_name", None)
        ctx.pop("pending_create_dataset_task_type", None)
        session.context = ctx
        return {"name": pending_name, "task_type": pending_tt}

    return None


async def _create_dataset_after_confirm(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    *,
    name: str,
    task_type: str,
    tool_events: list[dict[str, Any]],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """用户确认后直接创建数据集。"""

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    _emit({"type": "status", "phase": "tool", "text": "正在创建数据集…", "tool": "create_dataset", "label": "创建数据集"})
    tip = "好的，正在创建数据集…"
    session.messages.append(_msg("assistant", tip, ui={"kind": "assistant"}))
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    tt = normalize_task_type(task_type or "detect")
    session.context = {
        **(session.context or {}),
        "create_dataset_approved": {"name": name, "task_type": tt},
    }
    args = json.dumps({"name": name, "task_type": tt}, ensure_ascii=False)
    exec_result = await agent_tools.execute_tool("create_dataset", args, db=db, user=user)
    ctx = dict(session.context or {})
    ctx.pop("create_dataset_approved", None)
    ctx.pop("pending_create_dataset_name", None)
    ctx.pop("pending_create_dataset_task_type", None)
    session.context = ctx

    label = agent_tools.TOOL_LABELS.get("create_dataset", "create_dataset")
    tool_events.append(
        {
            "tool": "create_dataset",
            "label": label,
            "ok": bool(exec_result.get("ok")),
            "summary": _tool_summary("create_dataset", exec_result),
        }
    )
    call_id = f"call_confirm_{uuid.uuid4().hex[:8]}"
    session.messages.append(
        _msg(
            "assistant",
            None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "create_dataset", "arguments": args},
                }
            ],
            ui={"kind": "assistant", "planned_tools": [{"name": "create_dataset", "label": label}]},
        )
    )
    session.messages.append(
        _msg(
            "tool",
            agent_tools.tool_content_for_llm(exec_result),
            tool_call_id=call_id,
            name="create_dataset",
            ui=_tool_ui("create_dataset", exec_result),
        )
    )
    ask, job, _stop = _apply_tool_side_effects(session, "create_dataset", exec_result)
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    if not exec_result.get("ok"):
        err = str(exec_result.get("error") or "创建数据集失败")
        return f"创建数据集失败：{err}", ask, job

    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    ds_name = data.get("name") or name
    ds_id = data.get("dataset_id")
    if data.get("exists"):
        final = (
            f"已存在同名数据集「{ds_name}」#{ds_id}。"
            "可直接点对话框左下角 + 或下方按钮上传图片。"
        )
    else:
        final = (
            f"已创建数据集「{ds_name}」#{ds_id}。"
            "请点击对话框左下角 + 或下方「在对话中上传图片」上传图片。"
        )
    return final, ask, job


def _parse_tool_args(arguments: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return args if isinstance(args, dict) else {}


def _args_fingerprint(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)


def _action_confirm_question(tool_name: str, args: dict[str, Any]) -> str:
    """生成通用动作确认文案。"""
    label = agent_tools.TOOL_LABELS.get(tool_name, tool_name)
    lines = [f"接下来要「{label}」，可以继续吗？"]
    if tool_name == "open_annotate":
        if args.get("dataset_name"):
            lines.append(f"- 数据集：{args.get('dataset_name')}")
        elif args.get("dataset_id") is not None:
            lines.append(f"- 数据集：#{args.get('dataset_id')}")
    elif tool_name == "guide_dataset_upload":
        if args.get("dataset_name"):
            lines.append(f"- 数据集：{args.get('dataset_name')}")
        elif args.get("dataset_id") is not None:
            lines.append(f"- 数据集：#{args.get('dataset_id')}")
    elif tool_name == "start_prelabel":
        if args.get("dataset_id") is not None:
            lines.append(f"- 数据集：#{args.get('dataset_id')}")
    elif tool_name == "run_sample_infer":
        if args.get("task_id") is not None:
            lines.append(f"- 训练任务：#{args.get('task_id')}")
        if args.get("model_id") is not None:
            lines.append(f"- 模型：#{args.get('model_id')}")
    elif tool_name == "export_model_onnx":
        if args.get("model_id") is not None:
            lines.append(f"- 模型：#{args.get('model_id')}")
        if args.get("task_id") is not None:
            lines.append(f"- 任务：#{args.get('task_id')}")
    elif tool_name == "cancel_job":
        if args.get("job_id") is not None:
            lines.append(f"- 任务 Job：#{args.get('job_id')}")
    else:
        shown = 0
        for k, v in args.items():
            if shown >= 4:
                break
            if v is None or v == "":
                continue
            lines.append(f"- {k}：{v}")
            shown += 1
    lines.append("点「确认执行」开始；不想做就选「先不执行」。")
    return "\n".join(lines)


def _is_action_confirm_yes(text: str) -> bool:
    t = _normalize_confirm_text(text)
    if t in {_normalize_confirm_text(x) for x in _ACTION_CONFIRM_YES}:
        return True
    return ("确认" in t and "执行" in t) or t in {"执行", "开始执行"}


def _is_action_confirm_no(text: str) -> bool:
    t = _normalize_confirm_text(text)
    return t in {_normalize_confirm_text(x) for x in _ACTION_CONFIRM_NO} or (
        ("不" in t and ("执行" in t or "做" in t)) or t in {"取消", "算了"}
    )


def _gate_generic_action_result(
    session: AgentChatSessionOut,
    tool_name: str,
    arguments: str,
) -> dict[str, Any] | None:
    """通用写操作硬闸：未确认则拦截。"""
    if tool_name not in _GENERIC_CONFIRM_TOOLS:
        return None
    args = _parse_tool_args(arguments)
    fp = _args_fingerprint(args)
    approved = (session.context or {}).get("action_approved")
    if isinstance(approved, dict):
        if approved.get("tool") == tool_name and str(approved.get("fingerprint") or "") == fp:
            return None

    label = agent_tools.TOOL_LABELS.get(tool_name, tool_name)
    question = _action_confirm_question(tool_name, args)
    session.context = {
        **(session.context or {}),
        "pending_action": {
            "tool": tool_name,
            "arguments": args,
            "fingerprint": fp,
            "label": label,
        },
    }
    return {
        "ok": True,
        "data": {
            "need_confirm": True,
            "wait_for_user": True,
            "kind": "action_confirm",
            "question": question,
            "options": list(_ACTION_CONFIRM_OPTIONS),
            "tool": tool_name,
            "label": label,
            "message": f"尚未获得用户确认，未执行「{label}」。",
        },
    }


def _consume_action_confirm_reply(
    session: AgentChatSessionOut, user_text: str
) -> str | None:
    """
    处理通用动作确认。
    返回 'yes' | 'no' | None
    """
    ctx = dict(session.context or {})
    pending = ctx.get("pending_action")
    if not isinstance(pending, dict) or not pending.get("tool"):
        return None

    if _is_action_confirm_no(user_text):
        ctx.pop("pending_action", None)
        ctx.pop("action_approved", None)
        session.context = ctx
        return "no"

    if _is_action_confirm_yes(user_text):
        ctx["action_approved"] = {
            "tool": pending.get("tool"),
            "fingerprint": pending.get("fingerprint"),
            "arguments": pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {},
            "label": pending.get("label"),
        }
        ctx.pop("pending_action", None)
        session.context = ctx
        return "yes"

    return None


async def _execute_action_after_confirm(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    *,
    tool_events: list[dict[str, Any]],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """用户确认后执行 pending 的通用动作。"""

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    approved = (session.context or {}).get("action_approved")
    if not isinstance(approved, dict) or not approved.get("tool"):
        return "没有待执行的操作。", None, None

    tool_name = str(approved.get("tool"))
    args_obj = approved.get("arguments") if isinstance(approved.get("arguments"), dict) else {}
    label = str(approved.get("label") or agent_tools.TOOL_LABELS.get(tool_name, tool_name))
    args = json.dumps(args_obj, ensure_ascii=False)

    _emit({"type": "status", "phase": "tool", "text": f"正在{label}…", "tool": tool_name, "label": label})
    tip = f"好的，正在{label}…"
    session.messages.append(_msg("assistant", tip, ui={"kind": "assistant"}))
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    # ensure/set 仍需满足训练准备闸
    if tool_name in {"ensure_train_task", "set_train_config"}:
        gated_prep = _gate_train_prep_tool(session, tool_name, args)
        if gated_prep is not None:
            ctx = dict(session.context or {})
            ctx.pop("action_approved", None)
            session.context = ctx
            ask, job, _stop = _apply_tool_side_effects(session, tool_name, gated_prep)
            session.messages.append(
                _msg(
                    "tool",
                    agent_tools.tool_content_for_llm(gated_prep),
                    name=tool_name,
                    ui=_tool_ui(tool_name, gated_prep),
                )
            )
            save_session(session)
            return str((gated_prep.get("data") or {}).get("question") or "请先确认是否进入训练准备。"), ask, job

    exec_result = await agent_tools.execute_tool(tool_name, args, db=db, user=user)
    ctx = dict(session.context or {})
    ctx.pop("action_approved", None)
    ctx.pop("pending_action", None)
    session.context = ctx

    tool_events.append(
        {
            "tool": tool_name,
            "label": label,
            "ok": bool(exec_result.get("ok")),
            "summary": _tool_summary(tool_name, exec_result),
        }
    )
    call_id = f"call_confirm_{uuid.uuid4().hex[:8]}"
    session.messages.append(
        _msg(
            "assistant",
            None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }
            ],
            ui={"kind": "assistant", "planned_tools": [{"name": tool_name, "label": label}]},
        )
    )
    session.messages.append(
        _msg(
            "tool",
            agent_tools.tool_content_for_llm(exec_result),
            tool_call_id=call_id,
            name=tool_name,
            ui=_tool_ui(tool_name, exec_result),
        )
    )
    ask, job, _stop = _apply_tool_side_effects(session, tool_name, exec_result)
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    if not exec_result.get("ok"):
        err = str(exec_result.get("error") or f"{label}失败")
        return f"{label}失败：{err}", ask, job

    summary = _tool_summary(tool_name, exec_result)
    return f"已执行「{label}」。{summary}", ask, job


def _inspect_report_question(data: dict[str, Any]) -> tuple[str, list[str]]:
    """根据 inspect 结果生成停顿确认文案与选项。"""
    name = data.get("name") or data.get("dataset_name") or "当前数据集"
    labeled = data.get("labeled_count")
    unlabeled = data.get("unlabeled_count")
    class_count = data.get("class_count")
    can_train = bool(data.get("can_train"))
    lines = [
        f"标注情况（{name}）：",
        f"- 已标注 {labeled if labeled is not None else '-'} 张，未标注 {unlabeled if unlabeled is not None else '-'} 张",
        f"- 类别数：{class_count if class_count is not None else '-'}",
    ]
    if can_train:
        lines.append("已经可以进入训练准备了。要继续吗？（还不会马上开训）")
        return "\n".join(lines), list(_INSPECT_PREP_OPTIONS_OK)
    lines.append("标注还不够，建议先补标或再上传一些图。")
    return "\n".join(lines), list(_INSPECT_PREP_OPTIONS_BLOCKED)


def _enrich_inspect_with_gate(exec_result: dict[str, Any]) -> dict[str, Any]:
    """inspect 成功后附加 need_confirm，强制人机停顿。"""
    if not exec_result.get("ok"):
        return exec_result
    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    if not data:
        return exec_result
    question, options = _inspect_report_question(data)
    data = {
        **data,
        "need_confirm": True,
        "wait_for_user": True,
        "question": question,
        "options": options,
        "message": data.get("message") or "请确认是否进入训练准备。",
    }
    return {**exec_result, "data": data}


def _train_prep_allowed(session: AgentChatSessionOut, dataset_id: int | None) -> bool:
    if dataset_id is None:
        return False
    approved = (session.context or {}).get("train_prep_approved_dataset_id")
    try:
        return int(approved) == int(dataset_id)
    except (TypeError, ValueError):
        return False


def _gate_train_prep_tool(
    session: AgentChatSessionOut,
    tool_name: str,
    arguments: str,
) -> dict[str, Any] | None:
    """ensure/set 在未经「开始准备训练」确认时拦截。"""
    if tool_name not in {"ensure_train_task", "set_train_config"}:
        return None
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    ds_id = args.get("dataset_id")
    if ds_id is None:
        ds_id = (session.context or {}).get("dataset_id")
    try:
        ds_id_i = int(ds_id) if ds_id is not None else None
    except (TypeError, ValueError):
        ds_id_i = None
    if _train_prep_allowed(session, ds_id_i):
        return None

    # 回落到当前上下文报告
    ctx = session.context or {}
    fake = {
        "name": ctx.get("dataset_name") or "当前数据集",
        "dataset_id": ds_id_i or ctx.get("dataset_id"),
        "labeled_count": ctx.get("labeled_count"),
        "unlabeled_count": ctx.get("unlabeled_count"),
        "class_count": ctx.get("class_count"),
        "can_train": ctx.get("can_train", True),
        "task_type": ctx.get("task_type"),
    }
    question, options = _inspect_report_question(fake)
    if ds_id_i is not None:
        session.context = {
            **ctx,
            "pending_inspect_prep_dataset_id": ds_id_i,
        }
    return {
        "ok": True,
        "data": {
            "need_confirm": True,
            "wait_for_user": True,
            "question": question,
            "options": options,
            "dataset_id": ds_id_i,
            "name": fake.get("name"),
            "can_train": fake.get("can_train"),
            "message": "请先确认是否进入训练准备，再配置/开训。",
            # 让 side effect 走 inspect 分支
            "_gate_as": "inspect_dataset",
        },
    }


def _consume_inspect_prep_reply(session: AgentChatSessionOut, user_text: str) -> str | None:
    """
    处理标注复查后的用户选择。
    返回 'prep' | 'annotate' | None
    """
    ctx = dict(session.context or {})
    pending = ctx.get("pending_inspect_prep_dataset_id")
    if pending is None:
        return None
    try:
        ds_id = int(pending)
    except (TypeError, ValueError):
        ctx.pop("pending_inspect_prep_dataset_id", None)
        session.context = ctx
        return None

    if _is_inspect_prep_no(user_text):
        ctx.pop("pending_inspect_prep_dataset_id", None)
        ctx.pop("train_prep_approved_dataset_id", None)
        session.context = ctx
        return "annotate"

    if _is_inspect_prep_yes(user_text):
        ctx["train_prep_approved_dataset_id"] = ds_id
        ctx.pop("pending_inspect_prep_dataset_id", None)
        session.context = ctx
        return "prep"

    return None


def _gate_start_eval_result(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    arguments: str,
) -> dict[str, Any] | None:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict) or args.get("task_id") is None:
        return {"ok": False, "error": "start_eval 需要 task_id", "data": {}}
    task_id = int(args["task_id"])
    approved = (session.context or {}).get("eval_approved_task_id")
    try:
        approved_id = int(approved) if approved is not None else 0
    except (TypeError, ValueError):
        approved_id = 0
    if approved_id == task_id:
        return None

    task_name = ""
    try:
        task = agent_train_ops.get_owned_task(db, user, task_id)
        task_name = task.name
    except Exception:  # noqa: BLE001
        task_name = f"#{task_id}"

    session.context = {
        **(session.context or {}),
        "pending_eval_confirm_task_id": task_id,
        "task_id": task_id,
    }
    question = (
        f"训练任务「{task_name}」可以评估了。\n"
        "现在就评估吗？点「确认评估」开始，或选「先不评估」。"
    )
    return {
        "ok": True,
        "data": {
            "need_confirm": True,
            "wait_for_user": True,
            "question": question,
            "options": list(_EVAL_CONFIRM_OPTIONS),
            "task_id": task_id,
            "message": "尚未获得用户确认，未启动评估。",
        },
    }


def _consume_eval_confirm_reply(session: AgentChatSessionOut, user_text: str) -> int | None:
    ctx = dict(session.context or {})
    pending = ctx.get("pending_eval_confirm_task_id")
    if pending is None:
        return None
    try:
        task_id = int(pending)
    except (TypeError, ValueError):
        ctx.pop("pending_eval_confirm_task_id", None)
        session.context = ctx
        return None

    if _is_eval_confirm_no(user_text):
        ctx.pop("pending_eval_confirm_task_id", None)
        ctx.pop("eval_approved_task_id", None)
        session.context = ctx
        return None

    if _is_eval_confirm_yes(user_text):
        ctx["eval_approved_task_id"] = task_id
        ctx.pop("pending_eval_confirm_task_id", None)
        session.context = ctx
        return task_id
    return None


async def _start_eval_after_confirm(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    task_id: int,
    *,
    tool_events: list[dict[str, Any]],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    _emit({"type": "status", "phase": "tool", "text": "正在启动评估…", "tool": "start_eval", "label": "开始评估"})
    session.messages.append(_msg("assistant", "好的，正在启动评估…", ui={"kind": "assistant"}))
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    session.context = {**(session.context or {}), "eval_approved_task_id": task_id}
    args = json.dumps({"task_id": task_id}, ensure_ascii=False)
    exec_result = await agent_tools.execute_tool("start_eval", args, db=db, user=user)
    ctx = dict(session.context or {})
    ctx.pop("eval_approved_task_id", None)
    ctx.pop("pending_eval_confirm_task_id", None)
    session.context = ctx

    label = agent_tools.TOOL_LABELS.get("start_eval", "start_eval")
    tool_events.append(
        {
            "tool": "start_eval",
            "label": label,
            "ok": bool(exec_result.get("ok")),
            "summary": _tool_summary("start_eval", exec_result),
        }
    )
    call_id = f"call_eval_{uuid.uuid4().hex[:8]}"
    session.messages.append(
        _msg(
            "assistant",
            None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "start_eval", "arguments": args},
                }
            ],
            ui={"kind": "assistant", "planned_tools": [{"name": "start_eval", "label": label}]},
        )
    )
    session.messages.append(
        _msg(
            "tool",
            agent_tools.tool_content_for_llm(exec_result),
            tool_call_id=call_id,
            name="start_eval",
            ui=_tool_ui("start_eval", exec_result),
        )
    )
    ask, job, _stop = _apply_tool_side_effects(session, "start_eval", exec_result)
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})
    if not exec_result.get("ok"):
        return f"启动评估失败：{exec_result.get('error') or '未知错误'}", ask, job
    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    return f"已启动评估 Job #{data.get('job_id')}，可在下方查看进度。", ask, job


def _build_train_plan(db: Session, user: User, session: AgentChatSessionOut, task_id: int) -> dict[str, Any]:
    """汇总开训确认卡片所需信息。"""
    task = agent_train_ops.get_owned_task(db, user, task_id)
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    cfg: dict[str, Any] = {}
    try:
        cfg = json.loads(task.config_json or "{}")
        if not isinstance(cfg, dict):
            cfg = {}
    except json.JSONDecodeError:
        cfg = {}
    ctx_cfg = (session.context or {}).get("train_config")
    if isinstance(ctx_cfg, dict) and ctx_cfg:
        cfg = {**cfg, **ctx_cfg}

    tt = (ds.task_type if ds else None) or (session.context or {}).get("task_type") or "detect"
    labeled = None
    try:
        if ds is not None:
            info = dataset_prelabel.analyze_prelabel(Path(ds.path), ds.task_type or "detect")
            labeled = int(info.get("labeled_count") or 0)
    except Exception:  # noqa: BLE001
        labeled = (session.context or {}).get("labeled_count")

    return {
        "task_id": task.id,
        "task_name": task.name,
        "dataset_id": int(ds.id) if ds else None,
        "dataset_name": ds.name if ds else (session.context or {}).get("dataset_name"),
        "task_type": tt,
        "task_type_label": TASK_TYPE_LABELS.get(str(tt), str(tt)),
        "epochs": cfg.get("epochs"),
        "batch": cfg.get("batch"),
        "device": cfg.get("device") or "cpu",
        "pretrained_weight": cfg.get("pretrained_weight") or "（未指定，将使用默认）",
        "imgsz": cfg.get("imgsz"),
        "labeled_count": labeled,
    }


def _train_confirm_question(plan: dict[str, Any]) -> str:
    lines = [
        "准备开训，请确认方案：",
        f"- 数据集：{plan.get('dataset_name') or '-'}",
        f"- 类型：{plan.get('task_type_label') or plan.get('task_type') or '-'}",
        f"- 权重：{plan.get('pretrained_weight')}",
        f"- 轮数：{plan.get('epochs') if plan.get('epochs') is not None else '-'}",
        f"- batch：{plan.get('batch') if plan.get('batch') is not None else '-'}",
        f"- 设备：{plan.get('device') or 'cpu'}",
    ]
    if plan.get("imgsz") is not None:
        lines.append(f"- 输入尺寸：{plan.get('imgsz')}")
    if plan.get("labeled_count") is not None:
        lines.append(f"- 已标注：{plan.get('labeled_count')} 张")
    lines.append("没问题点「确认开训」；要调整选「我想改参数」。")
    return "\n".join(lines)


def _gate_start_train_result(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    arguments: str,
) -> dict[str, Any] | None:
    """
    若尚未确认开训，返回 need_confirm 结果（不真正开训）；
    已确认则返回 None，由调用方继续执行真实 start_train。
    """
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict) or args.get("task_id") is None:
        return {
            "ok": False,
            "error": "start_train 需要 task_id",
            "data": {},
        }
    task_id = int(args["task_id"])
    approved = (session.context or {}).get("train_start_approved_task_id")
    try:
        approved_id = int(approved) if approved is not None else 0
    except (TypeError, ValueError):
        approved_id = 0
    if approved_id == task_id:
        return None

    try:
        plan = _build_train_plan(db, user, session, task_id)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"无法准备训练确认：{e}", "data": {}}

    session.context = {
        **(session.context or {}),
        "pending_train_confirm_task_id": task_id,
        "task_id": task_id,
        "train_config": {
            "epochs": plan.get("epochs"),
            "batch": plan.get("batch"),
            "device": plan.get("device"),
            "pretrained_weight": plan.get("pretrained_weight"),
            "imgsz": plan.get("imgsz"),
        },
    }
    question = _train_confirm_question(plan)
    return {
        "ok": True,
        "data": {
            "need_confirm": True,
            "wait_for_user": True,
            "question": question,
            "options": list(_TRAIN_CONFIRM_OPTIONS),
            "task_id": task_id,
            "train_plan": plan,
            "message": "尚未获得用户确认，未启动训练。",
        },
    }


def _consume_train_confirm_reply(session: AgentChatSessionOut, user_text: str) -> int | None:
    """
    处理用户对开训确认的回复。
    返回 task_id 表示用户刚确认开训；否则返回 None。
    """
    ctx = dict(session.context or {})
    pending = ctx.get("pending_train_confirm_task_id")
    if pending is None:
        return None
    try:
        task_id = int(pending)
    except (TypeError, ValueError):
        ctx.pop("pending_train_confirm_task_id", None)
        session.context = ctx
        return None

    if _is_train_confirm_change(user_text):
        ctx.pop("pending_train_confirm_task_id", None)
        ctx.pop("train_start_approved_task_id", None)
        session.context = ctx
        return None

    if _is_train_confirm_yes(user_text):
        ctx["train_start_approved_task_id"] = task_id
        ctx.pop("pending_train_confirm_task_id", None)
        session.context = ctx
        return task_id

    # 其它回复：保留 pending，方便用户稍后仍可点确认
    return None


async def _start_train_after_confirm(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    task_id: int,
    *,
    tool_events: list[dict[str, Any]],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """用户确认后直接开训，不依赖模型再次调用工具。"""

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    _emit({"type": "status", "phase": "tool", "text": "正在启动训练…", "tool": "start_train", "label": "开始训练"})
    tip = "好的，正在启动训练…"
    session.messages.append(_msg("assistant", tip, ui={"kind": "assistant"}))
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    # 一次性授权：真正调用前保持 approved；成功或失败后都清掉，避免误触复用
    session.context = {
        **(session.context or {}),
        "train_start_approved_task_id": task_id,
    }
    args = json.dumps({"task_id": task_id}, ensure_ascii=False)
    exec_result = await agent_tools.execute_tool("start_train", args, db=db, user=user)
    # 无论成败，消费掉批准令牌
    ctx = dict(session.context or {})
    ctx.pop("train_start_approved_task_id", None)
    ctx.pop("pending_train_confirm_task_id", None)
    session.context = ctx

    label = agent_tools.TOOL_LABELS.get("start_train", "start_train")
    tool_events.append(
        {
            "tool": "start_train",
            "label": label,
            "ok": bool(exec_result.get("ok")),
            "summary": _tool_summary("start_train", exec_result),
        }
    )
    # 伪造一次 assistant tool_call + tool 消息，保持对话轨迹完整
    call_id = f"call_confirm_{uuid.uuid4().hex[:8]}"
    session.messages.append(
        _msg(
            "assistant",
            None,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "start_train", "arguments": args},
                }
            ],
            ui={"kind": "assistant", "planned_tools": [{"name": "start_train", "label": label}]},
        )
    )
    session.messages.append(
        _msg(
            "tool",
            agent_tools.tool_content_for_llm(exec_result),
            tool_call_id=call_id,
            name="start_train",
            ui=_tool_ui("start_train", exec_result),
        )
    )
    ask, job, _stop = _apply_tool_side_effects(session, "start_train", exec_result)
    save_session(session)
    _emit({"type": "session", "session": session.model_dump(mode="json")})

    if not exec_result.get("ok"):
        err = str(exec_result.get("error") or "启动训练失败")
        return f"启动训练失败：{err}", ask, job

    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    job_id = data.get("job_id")
    final = f"训练已启动（Job #{job_id}）。进度在下方查看，需要时可以停止。"
    return final, ask, job


def _tool_ui(tool_name: str, exec_result: dict[str, Any]) -> dict[str, Any]:
    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    ok = bool(exec_result.get("ok"))
    ui: dict[str, Any] = {
        "kind": "tool_result",
        "tool": tool_name,
        "label": agent_tools.TOOL_LABELS.get(tool_name, tool_name),
        "ok": ok,
        "summary": _tool_summary(tool_name, exec_result),
    }
    # 等待确认的闸门结果：问题已在助手气泡展示，工具卡片对用户是噪音
    if data.get("need_confirm") and data.get("wait_for_user"):
        ui["hidden"] = True
        return ui
    # 可展开详情：失败优先错误信息；成功则给精简 JSON（去掉预览大字段）
    if not ok:
        ui["detail"] = str(exec_result.get("error") or "执行失败")
    elif data:
        safe = {
            k: ("[已省略]" if ("base64" in k.lower() or "preview" in k.lower()) else v)
            for k, v in data.items()
            if k != "actions"
        }
        try:
            detail = json.dumps(safe, ensure_ascii=False, indent=2)
            if len(detail) > 3500:
                detail = detail[:3500] + "\n…"
            # 与摘要几乎相同时不塞详情，避免无意义展开
            if detail.strip() and detail.strip() != str(ui["summary"]).strip():
                ui["detail"] = detail
        except (TypeError, ValueError):
            pass
    actions = data.get("actions")
    if isinstance(actions, list) and actions:
        ui["actions"] = actions
    if tool_name in {"start_train", "start_eval", "start_prelabel", "export_model_onnx"} and data.get("job_id"):
        ui["kind"] = "job_progress"
        ui["job_id"] = data.get("job_id")
        ui["job_type"] = data.get("job_type")
        ui["progress"] = data.get("progress")
        ui["status"] = data.get("status")
    if tool_name == "run_sample_infer" and data.get("has_preview") and data.get("preview_base64"):
        ui["kind"] = "infer_preview"
        ui["preview_base64"] = data.get("preview_base64")
        ui["detection_count"] = data.get("detection_count")
        ui["sample_image"] = data.get("sample_image")
        if isinstance(actions, list):
            ui["actions"] = actions
    return ui


async def _llm_tool_loop(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    *,
    tool_events: list[dict[str, Any]],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """执行多轮 tool calling，返回 (final_text, pending_ask, pending_job)。"""
    cfg = _llm_settings()
    base_url = cfg["base_url"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    timeout = cfg["timeout"]
    thinking_enabled = bool(cfg["thinking_enabled"])
    reasoning_effort = str(cfg["reasoning_effort"])
    final_text = ""
    pending_ask: dict[str, Any] | None = None
    pending_job: dict[str, Any] | None = None

    def _emit(event: dict[str, Any]) -> None:
        if on_progress:
            on_progress(event)

    def _check_cancel() -> None:
        if is_chat_cancelled(session.id):
            raise ChatTurnCancelled("用户已终止本轮思考")

    for _ in range(MAX_TOOL_ROUNDS):
        _check_cancel()
        _emit({"type": "status", "phase": "thinking", "text": "正在思考…"})
        # 真正的 token 流式：边收边推 SSE（思考 / 回答增量）
        def _on_delta(d: dict[str, Any]) -> None:
            kind = str(d.get("kind") or "")
            delta = str(d.get("delta") or "")
            if not delta:
                return
            if kind == "reasoning":
                _emit({"type": "thinking_delta", "delta": delta})
            elif kind == "content":
                _emit({"type": "answer_delta", "delta": delta})

        try:
            result = await asyncio.to_thread(
                chat_completion_exchange_stream,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=_to_openai_messages(session, echo_reasoning=thinking_enabled),
                timeout=timeout,
                temperature=0.3,
                tools=agent_tools.openai_tool_schemas(),
                tool_choice="auto",
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
                on_delta=_on_delta,
                should_cancel=lambda: is_chat_cancelled(session.id),
            )
        except LLMStreamCancelled as e:
            raise ChatTurnCancelled(str(e)) from e
        _check_cancel()
        am = result.message
        openai_tool_calls: list[dict[str, Any]] = []
        for tc in am.tool_calls:
            openai_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
            )

        ui: dict[str, Any] = {"kind": "assistant"}
        if am.reasoning_content:
            ui["reasoning"] = am.reasoning_content
            # 完整思考文本（兼容旧前端）；增量已通过 thinking_delta 推送
            _emit({"type": "thinking", "text": am.reasoning_content})
        if openai_tool_calls:
            ui["planned_tools"] = [
                {
                    "name": tc.name,
                    "label": agent_tools.TOOL_LABELS.get(tc.name, tc.name),
                }
                for tc in am.tool_calls
            ]

        session.messages.append(
            _msg(
                "assistant",
                am.content,
                tool_calls=openai_tool_calls,
                ui=ui,
            )
        )
        save_session(session)
        _emit({"type": "session", "session": session.model_dump(mode="json")})

        if not am.tool_calls:
            final_text = am.content or ""
            break

        tool_names = [agent_tools.TOOL_LABELS.get(tc.name, tc.name) for tc in am.tool_calls]
        _emit(
            {
                "type": "status",
                "phase": "tools",
                "text": "准备调用：" + "、".join(tool_names),
                "tools": tool_names,
            }
        )

        stop_loop = False
        for idx, tc in enumerate(am.tool_calls):
            _check_cancel()
            label = agent_tools.TOOL_LABELS.get(tc.name, tc.name)
            _emit({"type": "status", "phase": "tool", "text": f"正在调用：{label}", "tool": tc.name, "label": label})
            if tc.name == "start_train":
                gated = _gate_start_train_result(db, user, session, tc.arguments)
                if gated is not None:
                    exec_result = gated
                else:
                    exec_result = await agent_tools.execute_tool(
                        tc.name, tc.arguments, db=db, user=user
                    )
                    ctx = dict(session.context or {})
                    ctx.pop("train_start_approved_task_id", None)
                    ctx.pop("pending_train_confirm_task_id", None)
                    session.context = ctx
            elif tc.name == "start_eval":
                gated = _gate_start_eval_result(db, user, session, tc.arguments)
                if gated is not None:
                    exec_result = gated
                else:
                    exec_result = await agent_tools.execute_tool(
                        tc.name, tc.arguments, db=db, user=user
                    )
                    ctx = dict(session.context or {})
                    ctx.pop("eval_approved_task_id", None)
                    ctx.pop("pending_eval_confirm_task_id", None)
                    session.context = ctx
            elif tc.name == "create_dataset":
                gated = _gate_create_dataset_result(session, tc.arguments)
                if gated is not None:
                    exec_result = gated
                else:
                    exec_result = await agent_tools.execute_tool(
                        tc.name, tc.arguments, db=db, user=user
                    )
                    ctx = dict(session.context or {})
                    ctx.pop("create_dataset_approved", None)
                    ctx.pop("pending_create_dataset_name", None)
                    ctx.pop("pending_create_dataset_task_type", None)
                    session.context = ctx
            elif tc.name in {"ensure_train_task", "set_train_config"}:
                # 仅过「开始准备训练」闸；通过后直接执行（开训另有专用确认）
                gated = _gate_train_prep_tool(session, tc.name, tc.arguments)
                if gated is not None:
                    exec_result = gated
                else:
                    exec_result = await agent_tools.execute_tool(
                        tc.name, tc.arguments, db=db, user=user
                    )
            elif tc.name in _GENERIC_CONFIRM_TOOLS:
                gated_action = _gate_generic_action_result(session, tc.name, tc.arguments)
                if gated_action is not None:
                    exec_result = gated_action
                else:
                    exec_result = await agent_tools.execute_tool(
                        tc.name, tc.arguments, db=db, user=user
                    )
                    ctx = dict(session.context or {})
                    ctx.pop("action_approved", None)
                    ctx.pop("pending_action", None)
                    session.context = ctx
            else:
                exec_result = await agent_tools.execute_tool(tc.name, tc.arguments, db=db, user=user)
                if tc.name == "inspect_dataset":
                    exec_result = _enrich_inspect_with_gate(exec_result)
            tool_events.append(
                {
                    "tool": tc.name,
                    "label": label,
                    "ok": bool(exec_result.get("ok")),
                    "summary": _tool_summary(tc.name, exec_result),
                }
            )
            session.messages.append(
                _msg(
                    "tool",
                    agent_tools.tool_content_for_llm(exec_result),
                    tool_call_id=tc.id,
                    name=tc.name,
                    ui=_tool_ui(tc.name, exec_result),
                )
            )
            ask, job, stop = _apply_tool_side_effects(session, tc.name, exec_result)
            if ask:
                pending_ask = ask
            if job:
                pending_job = job
            if stop:
                stop_loop = True
            save_session(session)
            _emit({"type": "session", "session": session.model_dump(mode="json")})
            if stop_loop:
                # 同一轮剩余 tool_calls 必须补齐回复，否则下次请求会 400
                rest = am.tool_calls[idx + 1 :]
                if rest:
                    _append_skipped_tool_messages(
                        session,
                        rest,
                        reason="已暂停等待用户确认，本轮其余工具未执行。",
                        tool_events=tool_events,
                    )
                    save_session(session)
                    _emit({"type": "session", "session": session.model_dump(mode="json")})
                break

        if stop_loop:
            if pending_ask:
                final_text = (am.content or "").strip() or str(pending_ask.get("question") or "")
            elif pending_job:
                final_text = (am.content or "").strip() or (
                    f"已启动 {pending_job.get('job_type')} Job #{pending_job.get('job_id')}，正在执行…"
                )
            break
    else:
        final_text = final_text or "本轮工具调用次数已达上限，请再发一条消息继续。"

    return final_text, pending_ask, pending_job


def _finalize_turn_state(
    session: AgentChatSessionOut,
    *,
    final_text: str,
    pending_ask: dict[str, Any] | None,
    pending_job: dict[str, Any] | None,
) -> str:
    """根据 pending 更新会话状态，必要时补 ask_user 气泡；返回最终助手文本。"""
    text = final_text
    if pending_job:
        session.pending_job = pending_job
        session.status = "waiting_job"
    elif pending_ask:
        session.pending_ask = pending_ask
        session.status = "waiting_user"
        q = str(pending_ask.get("question") or "").strip()
        if q and not any(
            m.role == "assistant" and (m.content or "").strip() == q for m in session.messages[-4:]
        ):
            session.messages.append(
                _msg(
                    "assistant",
                    q,
                    ui={"kind": "ask_user", "options": pending_ask.get("options") or []},
                )
            )
            text = q
    else:
        if session.pending_job:
            session.status = "waiting_job"
        else:
            session.status = "idle"
    return text


async def run_chat_turn(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    user_text: str,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> AgentChatTurnOut:
    """处理用户一轮输入：追加消息 → 多轮 tool calling → 返回更新后的会话。"""
    text = (user_text or "").strip()
    if not text:
        raise ValueError("消息不能为空")

    clear_chat_cancel(session.id)
    session.status = "running"
    session.pending_ask = None
    # 用户主动发话时，不清除 pending_job（可能仍在跑）；仅清除等待确认
    _maybe_update_title(session, text)
    session.messages.append(_msg("user", text))
    confirmed_tid = _consume_train_confirm_reply(session, text)
    eval_tid = None if confirmed_tid is not None else _consume_eval_confirm_reply(session, text)
    create_ds = None
    if confirmed_tid is None and eval_tid is None:
        create_ds = _consume_create_dataset_confirm_reply(session, text)
    action_decision = None
    if confirmed_tid is None and eval_tid is None and create_ds is None:
        action_decision = _consume_action_confirm_reply(session, text)
    inspect_decision = None
    if (
        confirmed_tid is None
        and eval_tid is None
        and create_ds is None
        and action_decision is None
    ):
        inspect_decision = _consume_inspect_prep_reply(session, text)
    save_session(session)
    if on_progress:
        on_progress({"type": "status", "phase": "thinking", "text": "正在思考…"})
        on_progress({"type": "session", "session": session.model_dump(mode="json")})

    tool_events: list[dict[str, Any]] = []
    try:
        # 方案 A：用户点「确认开训」后直接启动，不再依赖模型二次调用
        if confirmed_tid is not None:
            final_text, pending_ask, pending_job = await _start_train_after_confirm(
                db,
                user,
                session,
                confirmed_tid,
                tool_events=tool_events,
                on_progress=on_progress,
            )
        elif eval_tid is not None:
            final_text, pending_ask, pending_job = await _start_eval_after_confirm(
                db,
                user,
                session,
                eval_tid,
                tool_events=tool_events,
                on_progress=on_progress,
            )
        elif isinstance(create_ds, dict):
            final_text, pending_ask, pending_job = await _create_dataset_after_confirm(
                db,
                user,
                session,
                name=str(create_ds.get("name") or ""),
                task_type=str(create_ds.get("task_type") or "detect"),
                tool_events=tool_events,
                on_progress=on_progress,
            )
            # 创建成功后继续让模型引导上传/标注
            if pending_ask is None and pending_job is None:
                cont_text, pending_ask, pending_job = await _llm_tool_loop(
                    db, user, session, tool_events=tool_events, on_progress=on_progress
                )
                if cont_text:
                    final_text = cont_text
        elif action_decision == "yes":
            final_text, pending_ask, pending_job = await _execute_action_after_confirm(
                db,
                user,
                session,
                tool_events=tool_events,
                on_progress=on_progress,
            )
            # 中间步骤确认后继续推进（例如标注/预标注之后的下一步；开训仍会再确认）
            if pending_ask is None and pending_job is None:
                cont_text, pending_ask, pending_job = await _llm_tool_loop(
                    db, user, session, tool_events=tool_events, on_progress=on_progress
                )
                if cont_text:
                    final_text = cont_text
        else:
            if create_ds == "change":
                tip = (
                    "好的，请告诉我你想用的数据集名称，以及任务类型"
                    "（目标检测 detect / 实例分割 segment / 姿态估计 pose）。"
                    "确认后再创建。"
                )
                session.messages.append(_msg("assistant", tip, ui={"kind": "assistant"}))
                save_session(session)
                if on_progress:
                    on_progress({"type": "session", "session": session.model_dump(mode="json")})
                final_text = tip
                pending_ask, pending_job = None, None
            elif action_decision == "no":
                tip = "好的，已取消本次操作。你可以直接说下一步想做什么。"
                session.messages.append(_msg("assistant", tip, ui={"kind": "assistant"}))
                save_session(session)
                if on_progress:
                    on_progress({"type": "session", "session": session.model_dump(mode="json")})
                final_text = tip
                pending_ask, pending_job = None, None
            elif inspect_decision == "annotate":
                # 用户选择继续标注：助手气泡带「去标注」按钮（不要注入孤儿 tool 消息）
                tip = (
                    "好的，我们先继续完善标注。"
                    "你可点击下方「去标注」，标完后点「标完了，继续对话」。"
                )
                ds_id = (session.context or {}).get("dataset_id")
                tt = (session.context or {}).get("task_type") or "detect"
                ui: dict[str, Any] = {"kind": "assistant"}
                if ds_id:
                    ui["actions"] = [
                        {
                            "type": "annotate",
                            "label": "去标注",
                            "dataset_id": int(ds_id),
                            "task_type": tt,
                        }
                    ]
                session.messages.append(_msg("assistant", tip, ui=ui))
                save_session(session)
                if on_progress:
                    on_progress({"type": "session", "session": session.model_dump(mode="json")})
                final_text = tip
                pending_ask, pending_job = None, None
            else:
                # prep 确认或普通对话：走模型；prep 时模型应继续 ensure/set
                final_text, pending_ask, pending_job = await _llm_tool_loop(
                    db, user, session, tool_events=tool_events, on_progress=on_progress
                )
        final_text = _finalize_turn_state(
            session, final_text=final_text, pending_ask=pending_ask, pending_job=pending_job
        )
        save_session(session)
        out = AgentChatTurnOut(session=session, assistant_text=final_text, tool_events=tool_events)
        if on_progress:
            on_progress({"type": "done", "turn": out.model_dump(mode="json")})
        return out
    except ChatTurnCancelled:
        session.status = "idle"
        tip = "已停止生成。你可以继续发消息；若后台仍有训练 Job，可用「停止训练」终止。"
        session.messages.append(_msg("assistant", tip, ui={"kind": "cancelled"}))
        save_session(session)
        out = AgentChatTurnOut(session=session, assistant_text=tip, tool_events=tool_events)
        if on_progress:
            on_progress({"type": "cancelled", "session": session.model_dump(mode="json")})
            on_progress({"type": "done", "turn": out.model_dump(mode="json")})
        return out
    except Exception as e:  # noqa: BLE001
        session.status = "idle"
        err = str(e)
        session.messages.append(_msg("assistant", f"调用失败：{err}", ui={"kind": "error"}))
        save_session(session)
        if on_progress:
            on_progress({"type": "error", "message": err, "session": session.model_dump(mode="json")})
        raise
    finally:
        clear_chat_cancel(session.id)


async def iter_chat_turn_events(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    user_text: str,
) -> AsyncIterator[dict[str, Any]]:
    """流式产出本轮进度事件（供 SSE）。"""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_progress(ev: dict[str, Any]) -> None:
        # LLM 在线程中回调，必须线程安全地投递到事件循环
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    async def _runner() -> None:
        try:
            await run_chat_turn(db, user, session, user_text, on_progress=_on_progress)
        except Exception as e:  # noqa: BLE001
            # run_chat_turn 已 emit error；取消不算错误
            if not isinstance(e, (ValueError, RuntimeError, ChatTurnCancelled)):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(_runner())
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    finally:
        # 客户端断开流时也请求取消，避免后台空转
        request_chat_cancel(session.id)
        await task


async def continue_after_job(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> AgentChatTurnOut:
    """
    前端轮询到 pending_job 结束后调用：写入 Job 结果，让模型继续（可评估/推理/总结）。
    """
    pending = session.pending_job or {}
    job_id = pending.get("job_id") or (session.context or {}).get("active_job_id")
    if not job_id:
        # 幂等：已续跑过 / 本地状态过期时不要 400 刷屏
        if session.status == "waiting_job":
            session.status = "idle"
            save_session(session)
        return AgentChatTurnOut(session=session, assistant_text="", tool_events=[])

    status = agent_train_ops.get_job_status(db, user, job_id=int(job_id))
    if not status.get("done"):
        # 尚未结束：只回写进度信息
        session.messages.append(
            _msg(
                "assistant",
                f"Job #{job_id} 仍在进行：{status.get('status')} · {int(float(status.get('progress') or 0))}% · {status.get('message')}",
                ui={
                    "kind": "job_progress",
                    "job_id": job_id,
                    "job_type": status.get("job_type"),
                    "progress": status.get("progress"),
                    "status": status.get("status"),
                    "summary": status.get("message"),
                },
            )
        )
        session.status = "waiting_job"
        save_session(session)
        return AgentChatTurnOut(session=session, assistant_text="", tool_events=[])

    job_status = str(status.get("status") or "")
    # 结束：写入一条系统提示，再让模型继续（取消也要收尾，避免前端空转）
    if job_status == "cancelled":
        tip = (
            f"[系统] Job #{job_id}（{status.get('job_type')}）已停止："
            f"message={status.get('message')}。请用中文简短告知用户训练已取消，并询问是否调整配置后重试。"
        )
    else:
        tip = (
            f"[系统] Job #{job_id}（{status.get('job_type')}）已结束：status={status.get('status')}，"
            f"message={status.get('message')}，metrics={json.dumps(status.get('metrics_summary') or {}, ensure_ascii=False)}。"
            f"请根据结果用中文简要总结，并询问用户是否进行评估或抽样推理；不要擅自 start_eval。"
            f"若失败请说明原因与建议。"
        )
    session.messages.append(
        _msg(
            "user",
            tip,
            ui={"kind": "system_job_done", "job_id": job_id, "status": job_status},
        )
    )
    session.pending_job = None
    if session.context:
        session.context.pop("active_job_id", None)
        session.context.pop("active_job_type", None)
        session.context["last_job"] = status

    clear_chat_cancel(session.id)
    session.status = "running"
    save_session(session)
    if on_progress:
        on_progress({"type": "status", "phase": "thinking", "text": "任务已结束，正在总结…"})
        on_progress({"type": "session", "session": session.model_dump(mode="json")})

    tool_events: list[dict[str, Any]] = []
    try:
        final_text, pending_ask, pending_job = await _llm_tool_loop(
            db, user, session, tool_events=tool_events, on_progress=on_progress
        )
        final_text = _finalize_turn_state(
            session, final_text=final_text, pending_ask=pending_ask, pending_job=pending_job
        )
        save_session(session)
        out = AgentChatTurnOut(session=session, assistant_text=final_text, tool_events=tool_events)
        if on_progress:
            on_progress({"type": "done", "turn": out.model_dump(mode="json")})
        return out
    except ChatTurnCancelled:
        session.status = "idle"
        tip_msg = "已停止生成。训练 Job 已结束；你可以继续发消息。"
        session.messages.append(_msg("assistant", tip_msg, ui={"kind": "cancelled"}))
        save_session(session)
        out = AgentChatTurnOut(session=session, assistant_text=tip_msg, tool_events=tool_events)
        if on_progress:
            on_progress({"type": "cancelled", "session": session.model_dump(mode="json")})
            on_progress({"type": "done", "turn": out.model_dump(mode="json")})
        return out
    except Exception as e:  # noqa: BLE001
        session.status = "idle"
        # 即使续跑失败，也已清空 pending_job，避免前端死循环
        err = str(e)
        session.messages.append(_msg("assistant", f"Job 结束后继续失败：{err}", ui={"kind": "error"}))
        save_session(session)
        if on_progress:
            on_progress({"type": "error", "message": err, "session": session.model_dump(mode="json")})
        raise
    finally:
        clear_chat_cancel(session.id)


async def iter_continue_job_events(
    db: Session,
    user: User,
    session: AgentChatSessionOut,
) -> AsyncIterator[dict[str, Any]]:
    """流式产出 Job 续跑进度事件（供 SSE）。"""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_progress(ev: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    async def _runner() -> None:
        try:
            await continue_after_job(db, user, session, on_progress=_on_progress)
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, (ValueError, RuntimeError, ChatTurnCancelled)):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(_runner())
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    finally:
        request_chat_cancel(session.id)
        await task


def _tool_summary(name: str, exec_result: dict[str, Any]) -> str:
    if not exec_result.get("ok"):
        return str(exec_result.get("error") or "执行失败")
    data = exec_result.get("data") if isinstance(exec_result.get("data"), dict) else {}
    if data.get("need_confirm") and data.get("kind") == "action_confirm":
        label = data.get("label") or agent_tools.TOOL_LABELS.get(name, name)
        return f"等待用户确认：{label}"
    if name == "list_datasets":
        groups = data.get("datasets") or []
        parts = [f"{g.get('task_type_label')}:{g.get('count')}个" for g in groups if isinstance(g, dict)]
        return "；".join(parts) if parts else "无数据集"
    if name == "inspect_dataset":
        if data.get("need_confirm"):
            return "标注复查完成，等待确认是否准备训练"
        return (
            f"{data.get('name')} 已标注{data.get('labeled_count')} "
            f"未标注{data.get('unlabeled_count')} 类别{data.get('class_count')} "
            f"{'可训练' if data.get('can_train') else '暂不可训练'}"
        )
    if name == "list_weights":
        return f"{data.get('task_type')} 共 {data.get('count')} 个权重"
    if name == "create_dataset":
        if data.get("need_confirm"):
            return "等待用户确认创建"
        return f"{'已创建' if data.get('created') else '已存在'} {data.get('name')} #{data.get('dataset_id')}"
    if name == "guide_dataset_upload":
        return f"引导上传 {data.get('name') or ''} #{data.get('dataset_id') or ''}".strip()
    if name == "open_annotate":
        return f"打开标注 {data.get('name')} #{data.get('dataset_id')}"
    if name == "start_prelabel":
        return f"预标注 Job #{data.get('job_id')} · {data.get('status')}"
    if name == "start_eval":
        if data.get("need_confirm"):
            return "等待用户确认评估"
        return f"评估 Job #{data.get('job_id')} · {data.get('status')}"
    if name == "ensure_train_task":
        if data.get("need_confirm"):
            return (
                "等待用户确认执行"
                if data.get("kind") == "action_confirm"
                else "请先确认是否进入训练准备"
            )
        return f"任务 #{data.get('task_id')} {data.get('task_name')}"
    if name == "set_train_config":
        if data.get("need_confirm"):
            return (
                "等待用户确认执行"
                if data.get("kind") == "action_confirm"
                else "请先确认是否进入训练准备"
            )
        cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
        return f"epochs={cfg.get('epochs')} batch={cfg.get('batch')} device={cfg.get('device')} weight={cfg.get('pretrained_weight')}"
    if name == "start_train":
        if data.get("need_confirm"):
            return "等待用户确认开训"
        return f"训练 Job #{data.get('job_id')} · {data.get('status')}"
    if name == "get_job_status":
        return f"Job #{data.get('job_id')} {data.get('status')} {int(float(data.get('progress') or 0))}%"
    if name == "run_sample_infer":
        return f"样例 {data.get('sample_image')} 检出 {data.get('detection_count')} · {data.get('usable_hint')}"
    if name == "list_models":
        return f"模型库 {data.get('count')} 个"
    if name == "export_model_onnx":
        if data.get("already_exists"):
            return f"ONNX 已存在 {data.get('onnx_name')}"
        return f"导出 Job #{data.get('job_id')} · {data.get('status')}"
    if name == "cancel_job":
        return str(data.get("message") or f"Job #{data.get('job_id')}")
    if name == "list_devices":
        return f"设备 {len(data.get('devices') or [])} 个 · GPU={data.get('gpu_count')}"
    if name == "ask_user":
        return str(data.get("question") or "等待用户确认")
    return "完成"
