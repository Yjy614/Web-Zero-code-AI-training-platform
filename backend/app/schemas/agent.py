"""AI 流程编排 Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


StepType = Literal[
    "ensure_task",
    "check_labels",
    "prelabel",
    "review_labels",
    "apply_config",
    "split",
    "train",
    "eval",
    "export",
    "summary",
]

StepStatus = Literal["pending", "ready", "running", "completed", "failed", "skipped"]


class AgentPlanRequest(BaseModel):
    """根据自然语言生成编排计划。"""

    prompt: str = Field(..., min_length=1, max_length=2000)
    task_type: str = Field(default="detect", description="detect | segment")
    dataset_id: int
    task_id: int | None = None
    # 可选覆盖（LLM/模板都会参考）
    epochs: int | None = Field(default=None, ge=1, le=500)
    include_eval: bool = True
    include_export: bool = True
    include_prelabel: bool = False


class AgentStepOut(BaseModel):
    id: str
    type: StepType
    title: str
    description: str
    """是否必须人工确认后才执行（全部步骤默认需确认，本字段强调门禁含义）。"""
    human_gate: bool = False
    status: StepStatus = "pending"
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    job_id: int | None = None


class AgentPlanOut(BaseModel):
    id: str
    owner_id: int
    prompt: str
    task_type: str
    dataset_id: int
    task_id: int | None = None
    summary: str
    source: str = Field(description="llm | template")
    status: Literal["draft", "running", "completed", "cancelled", "failed"] = "draft"
    current_index: int = 0
    steps: list[AgentStepOut] = Field(default_factory=list)
    """解析/体检提示（非阻断，供前端展示）。"""
    hints: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class AgentPlanPatchRequest(BaseModel):
    """微调尚未完成的计划：改配置、移除可选步骤。"""

    epochs: int | None = Field(default=None, ge=1, le=500)
    batch: int | None = Field(default=None, ge=1, le=128)
    device: str | None = Field(default=None, max_length=32)
    pretrained_weight: str | None = Field(default=None, max_length=128)
    remove_step_ids: list[str] = Field(default_factory=list)


class AgentRunStepRequest(BaseModel):
    """执行当前步骤（需前端明确点确认）。"""

    action: Literal["run", "skip"] = "run"


class AgentRunStepOut(BaseModel):
    plan: AgentPlanOut
    message: str = ""


# ---------- 对话式 Agent ----------

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