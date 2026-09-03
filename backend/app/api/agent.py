"""AI 流程编排 API。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.task_types import normalize_task_type
from app.models.dataset import Dataset
from app.models.train import Job
from app.models.user import User
from app.runners.mock_runner import get_runner
from app.schemas.agent import (
    AgentChatCreateRequest,
    AgentChatRenameRequest,
    AgentChatSendRequest,
    AgentChatSessionOut,
    AgentChatSessionSummary,
    AgentChatTurnOut,
    AgentPlanOut,
    AgentPlanPatchRequest,
    AgentPlanRequest,
    AgentRunStepOut,
    AgentRunStepRequest,
)
from app.services import agent_chat, agent_orchestrate

router = APIRouter(prefix="/agent", tags=["AI编排"])


def _get_dataset(db: Session, dataset_id: int, user: User) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "数据集不存在"})
    if user.role != "admin" and ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权使用该数据集"})
    return ds


def _load_owned_plan(user: User, plan_id: str) -> AgentPlanOut:
    plan = agent_orchestrate.load_plan(user.id, plan_id)
    if not plan and user.role == "admin":
        # 管理员不可跨用户扫盘；仅本用户计划
        pass
    if not plan:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "编排计划不存在"})
    if plan.owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问该计划"})
    return plan


@router.post("/plan", response_model=AgentPlanOut)
def create_agent_plan(
    body: AgentPlanRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentPlanOut:
    """根据自然语言与数据集生成逐步确认的编排计划。"""
    ds = _get_dataset(db, body.dataset_id, user)
    try:
        tt = normalize_task_type(body.task_type or ds.task_type or "detect")
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_task_type", "message": str(e)}) from e
    if (ds.task_type or "detect") != tt:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "task_type_mismatch",
                "message": f"数据集类型为 {ds.task_type}，与请求 {tt} 不一致",
            },
        )
    body.task_type = tt
    return agent_orchestrate.generate_plan(db, user, body, ds)


@router.get("/plans/{plan_id}", response_model=AgentPlanOut)
def get_agent_plan(plan_id: str, user: User = Depends(get_current_user)) -> AgentPlanOut:
    return _load_owned_plan(user, plan_id)


@router.post("/plans/{plan_id}/steps/run", response_model=AgentRunStepOut)
async def run_agent_step(
    plan_id: str,
    body: AgentRunStepRequest | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentRunStepOut:
    """确认并执行当前步骤（或跳过）。"""
    plan = _load_owned_plan(user, plan_id)
    action = (body.action if body else "run") or "run"
    runner = get_runner()

    async def submit_job(job_type: str, payload: dict) -> None:
        await runner.submit(job_type, payload)

    try:
        plan, message = await agent_orchestrate.run_current_step(
            db=db,
            user=user,
            plan=plan,
            action=action,
            submit_job=submit_job,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "step_failed", "message": str(e)}) from e
    return AgentRunStepOut(plan=plan, message=message)


@router.post("/plans/{plan_id}/steps/ack-job", response_model=AgentRunStepOut)
def ack_agent_job(
    plan_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentRunStepOut:
    """当前步骤关联的 Job 结束后，确认推进计划。"""
    plan = _load_owned_plan(user, plan_id)
    if plan.current_index < 0 or plan.current_index >= len(plan.steps):
        raise HTTPException(status_code=400, detail={"code": "no_step", "message": "没有当前步骤"})
    step = plan.steps[plan.current_index]
    if not step.job_id:
        raise HTTPException(status_code=400, detail={"code": "no_job", "message": "当前步骤没有关联 Job"})
    job = db.query(Job).filter(Job.id == step.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Job 不存在"})
    try:
        plan = agent_orchestrate.ack_job_step(plan, job)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "ack_failed", "message": str(e)}) from e
    return AgentRunStepOut(plan=plan, message=job.message or job.status)


@router.patch("/plans/{plan_id}", response_model=AgentPlanOut)
def patch_agent_plan(
    plan_id: str,
    body: AgentPlanPatchRequest,
    user: User = Depends(get_current_user),
) -> AgentPlanOut:
    """微调计划：改 epochs/batch/device/权重，或移除未执行的可选步骤。"""
    plan = _load_owned_plan(user, plan_id)
    try:
        return agent_orchestrate.patch_plan(
            plan,
            {
                "epochs": body.epochs,
                "batch": body.batch,
                "device": body.device,
                "pretrained_weight": body.pretrained_weight,
                "remove_step_ids": body.remove_step_ids,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "patch_failed", "message": str(e)}) from e


@router.post("/plans/{plan_id}/cancel", response_model=AgentPlanOut)
def cancel_agent_plan(plan_id: str, user: User = Depends(get_current_user)) -> AgentPlanOut:
    plan = _load_owned_plan(user, plan_id)
    return agent_orchestrate.cancel_plan(plan)


# ---------- 对话式 Agent ----------


def _load_owned_chat(user: User, session_id: str) -> AgentChatSessionOut:
    sess = agent_chat.load_session(user.id, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "会话不存在"})
    if sess.owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问该会话"})
    return sess


@router.get("/chat/sessions", response_model=list[AgentChatSessionSummary])
def list_chat_sessions(user: User = Depends(get_current_user)) -> list[AgentChatSessionSummary]:
    return agent_chat.list_sessions(user.id)


@router.post("/chat/sessions", response_model=AgentChatSessionOut)
def create_chat_session(
    body: AgentChatCreateRequest | None = None,
    user: User = Depends(get_current_user),
) -> AgentChatSessionOut:
    title = body.title if body else None
    return agent_chat.create_session(user, title=title)


@router.get("/chat/sessions/{session_id}", response_model=AgentChatSessionOut)
def get_chat_session(session_id: str, user: User = Depends(get_current_user)) -> AgentChatSessionOut:
    return _load_owned_chat(user, session_id)


@router.patch("/chat/sessions/{session_id}", response_model=AgentChatSessionOut)
def rename_chat_session(
    session_id: str,
    body: AgentChatRenameRequest,
    user: User = Depends(get_current_user),
) -> AgentChatSessionOut:
    """重命名对话标题。"""
    _load_owned_chat(user, session_id)
    try:
        return agent_chat.rename_session(user.id, session_id, body.title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "rename_failed", "message": str(e)}) from e


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: str, user: User = Depends(get_current_user)) -> dict:
    _load_owned_chat(user, session_id)
    agent_chat.delete_session(user.id, session_id)
    return {"ok": True}


@router.post("/chat/sessions/{session_id}/messages", response_model=AgentChatTurnOut)
async def send_chat_message(
    session_id: str,
    body: AgentChatSendRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentChatTurnOut:
    """发送用户消息，触发一轮思考 + 工具调用。"""
    sess = _load_owned_chat(user, session_id)
    try:
        return await agent_chat.run_chat_turn(db, user, sess, body.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "chat_failed", "message": str(e)}) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail={"code": "llm_failed", "message": str(e)}) from e


@router.post("/chat/sessions/{session_id}/messages/stream")
async def send_chat_message_stream(
    session_id: str,
    body: AgentChatSendRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """流式发送：先思考进度，再按需工具，最后完成（SSE）。"""
    sess = _load_owned_chat(user, session_id)

    async def event_gen():
        try:
            async for ev in agent_chat.iter_chat_turn_events(db, user, sess, body.message):
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except ValueError as e:
            err = json.dumps(
                {"type": "error", "message": str(e), "code": "chat_failed"},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n"
        except RuntimeError as e:
            err = json.dumps(
                {"type": "error", "message": str(e), "code": "llm_failed"},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/sessions/{session_id}/cancel")
def cancel_chat_turn(
    session_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """终止当前会话正在进行的思考/工具循环（不取消后台训练 Job）。"""
    _load_owned_chat(user, session_id)
    agent_chat.request_chat_cancel(session_id)
    return {"ok": True, "message": "已请求停止生成"}


@router.post("/chat/sessions/{session_id}/continue-job", response_model=AgentChatTurnOut)
async def continue_chat_after_job(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentChatTurnOut:
    """pending_job 结束后由前端调用，把结果回写并让 Agent 继续。"""
    sess = _load_owned_chat(user, session_id)
    try:
        return await agent_chat.continue_after_job(db, user, sess)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "job_continue_failed", "message": str(e)}) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail={"code": "llm_failed", "message": str(e)}) from e


@router.post("/chat/sessions/{session_id}/continue-job/stream")
async def continue_chat_after_job_stream(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Job 结束后流式续跑（SSE），与发消息流式体验一致。"""
    sess = _load_owned_chat(user, session_id)

    async def event_gen():
        try:
            async for ev in agent_chat.iter_continue_job_events(db, user, sess):
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except ValueError as e:
            err = json.dumps(
                {"type": "error", "message": str(e), "code": "job_continue_failed"},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n"
        except RuntimeError as e:
            err = json.dumps(
                {"type": "error", "message": str(e), "code": "llm_failed"},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
