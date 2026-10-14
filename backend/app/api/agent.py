"""对话式 AI Agent API。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.agent import (
    AgentChatCreateRequest,
    AgentChatRenameRequest,
    AgentChatSendRequest,
    AgentChatSessionOut,
    AgentChatSessionSummary,
    AgentChatTurnOut,
)
from app.services import agent_chat

router = APIRouter(prefix="/agent", tags=["AI Agent"])


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
