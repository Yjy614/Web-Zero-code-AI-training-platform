"""对话式 AI Agent Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ChatRole = Literal["user", "assistant", "tool", "system"]


class AgentChatMessageOut(BaseModel):
    id: str
    role: ChatRole
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    """前端展示元数据：kind=assistant|tool_result|ask_user|error 等。"""
    ui: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class AgentChatSessionOut(BaseModel):
    id: str
    owner_id: int
    title: str = "新对话"
    status: Literal["idle", "running", "waiting_user", "waiting_job"] = "idle"
    messages: list[AgentChatMessageOut] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    pending_ask: dict[str, Any] | None = None
    """进行中的异步 Job：{job_id, job_type, task_id}。"""
    pending_job: dict[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""


class AgentChatSessionSummary(BaseModel):
    id: str
    title: str
    updated_at: str = ""
    message_count: int = 0
    status: str = "idle"


class AgentChatCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class AgentChatRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class AgentChatSendRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class AgentChatTurnOut(BaseModel):
    session: AgentChatSessionOut
    assistant_text: str = ""
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
