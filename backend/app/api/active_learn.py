"""主动学习 API：模型库入口 → 筛难例 → 并入原集 → 准备续训。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.dataset import Dataset
from app.models.train import Job, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner
from app.schemas.task import JobOut
from app.services import model_artifacts
from app.services.active_learning import pipeline as al_pipeline
from app.services.active_learning import session_store
from app.services.runtime_settings import is_demo_mode

router = APIRouter(prefix="/active-learn", tags=["主动学习"])


class CreateSessionBody(BaseModel):
    model_id: int
    target_dataset_id: int | None = None


class RetrainBody(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    batch: int = Field(default=8, ge=1, le=64)
    device: str = "cpu"
    imgsz: int = Field(default=640, ge=320, le=1280)


def _job_out(job: Job) -> JobOut:
    try:
        result = json.loads(job.result_json or "{}")
    except json.JSONDecodeError:
        result = {}
    if not isinstance(result, dict):
        result = {}
    return JobOut(
        id=job.id,
        task_id=job.task_id,
        type=job.type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=result,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _ensure_task_for_dataset(db: Session, ds: Dataset) -> TrainTask:
    task = (
        db.query(TrainTask)
        .filter(TrainTask.dataset_id == ds.id)
        .order_by(TrainTask.id.desc())
        .first()
    )
    if task:
        return task
    task = TrainTask(
        name=f"{ds.name}_al",
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        status="draft",
        step=2,
        config_json="{}",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.get("/models/{model_id}/hint")
def model_al_hint(
    model_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """模型库卡片用：是否可主动学习；原数据集由模型训练关联锁定。"""
    from app.models.train import ModelRecord
    from app.services import dataset_storage
    from app.services.active_learning.staging import cleanup_orphaned_staging

    # 进入主动学习前顺带清悬空临时集
    try:
        cleanup_orphaned_staging(db, user)
    except Exception:  # noqa: BLE001
        pass

    row = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "模型不存在"})
    if user.role != "admin" and row.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问"})
    tt = (row.task_type or "detect").strip().lower()
    from app.services.active_learning.strategy import SUPPORTED_TASK_TYPES

    supported = tt in SUPPORTED_TASK_TYPES
    has_pt = model_artifacts.find_pt_path(row) is not None
    target_id = al_pipeline.default_target_dataset_id(db, row)
    target_name = None
    image_count = None
    class_count = None
    if target_id:
        ds = db.query(Dataset).filter(Dataset.id == target_id).first()
        if ds:
            target_name = ds.name
            try:
                classes = json.loads(ds.classes_json or "[]")
            except json.JSONDecodeError:
                classes = []
            class_count = len(classes) if isinstance(classes, list) else 0
            if ds.path:
                try:
                    image_count = dataset_storage.count_active_images(Path(ds.path))
                except Exception:  # noqa: BLE001
                    image_count = int(ds.image_count or 0)

    blocked = None
    if not supported:
        blocked = f"暂不支持任务类型：{tt}"
    elif not has_pt:
        blocked = "缺少 PT 权重"
    elif not target_id:
        blocked = "找不到该模型的原训练数据集（模型未关联训练任务或数据集已删除），无法开始主动学习"

    return {
        "model_id": row.id,
        "model_name": row.name,
        "task_type": tt,
        "supported": supported,
        "has_pt": has_pt,
        "can_start": blocked is None,
        "default_target_dataset_id": target_id,
        "default_target_dataset_name": target_name,
        "target_image_count": image_count,
        "target_class_count": class_count,
        "message": blocked,
    }


@router.post("/sessions")
def create_session(
    body: CreateSessionBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 入口来自模型卡：强制使用模型关联的原数据集，忽略客户端传入的其它集
    try:
        session = al_pipeline.create_al_session(
            db,
            user,
            model_id=body.model_id,
            target_dataset_id=None,
        )
    except (ValueError, FileNotFoundError, PermissionError) as e:
        raise HTTPException(status_code=400, detail={"code": "create_failed", "message": str(e)}) from e
    return {"session": session}


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
):
    try:
        session = session_store.load_session(user.id, session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(e)}) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": str(e)}) from e
    return {"session": session}


@router.post("/sessions/{session_id}/screen")
async def start_screen(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """启动筛图 Job。"""
    try:
        session = session_store.load_session(user.id, session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(e)}) from e

    staging = db.query(Dataset).filter(Dataset.id == int(session["staging_dataset_id"])).first()
    if not staging:
        raise HTTPException(status_code=400, detail={"code": "no_staging", "message": "临时数据集不存在"})
    root = Path(staging.path)
    from app.services import dataset_storage

    if not dataset_storage.list_images(root, include_removed=False):
        raise HTTPException(status_code=400, detail={"code": "no_images", "message": "请先上传本轮新图片"})

    # 刷新计数
    staging.image_count = dataset_storage.count_images(root)
    db.commit()

    task = _ensure_task_for_dataset(db, staging)
    running = (
        db.query(Job)
        .filter(
            Job.task_id == task.id,
            Job.type == "active_learn",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if running:
        raise HTTPException(status_code=400, detail={"code": "busy", "message": "已有筛图任务在进行中"})

    if not is_demo_mode():
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=400,
                detail={"code": "deps_missing", "message": "未安装 ultralytics"},
            ) from e

    job = Job(task_id=task.id, type="active_learn", status="pending", progress=0, message="排队中")
    db.add(job)
    db.commit()
    db.refresh(job)

    session["job_id"] = job.id
    session["status"] = "screening"
    session_store.save_session(user.id, session)

    await get_runner().submit(
        "active_learn",
        {"job_id": job.id, "session_id": session_id, "user_id": user.id},
    )
    return {"job": _job_out(job), "session": session}


@router.post("/sessions/{session_id}/merge")
def merge_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        result = al_pipeline.merge_staging_into_target(db, user, session_id)
    except (ValueError, FileNotFoundError, PermissionError) as e:
        raise HTTPException(status_code=400, detail={"code": "merge_failed", "message": str(e)}) from e
    return result


@router.post("/sessions/{session_id}/abandon")
def abandon_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """取消本轮主动学习并删除临时数据集。"""
    from app.services.active_learning.staging import abandon_session as abandon_al

    try:
        return abandon_al(db, user, session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(e)}) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": str(e)}) from e


@router.post("/cleanup-orphans")
def cleanup_orphans(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """清理无进行中会话引用的主动学习临时集（中断悬空）。"""
    from app.services.active_learning.staging import cleanup_orphaned_staging

    return cleanup_orphaned_staging(db, user)


@router.post("/sessions/{session_id}/prepare-retrain")
def prepare_retrain(
    session_id: str,
    body: RetrainBody | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    body = body or RetrainBody()
    try:
        result = al_pipeline.prepare_retrain_task(
            db,
            user,
            session_id,
            epochs=body.epochs,
            batch=body.batch,
            device=body.device,
            imgsz=body.imgsz,
        )
    except (ValueError, FileNotFoundError, PermissionError) as e:
        raise HTTPException(status_code=400, detail={"code": "retrain_failed", "message": str(e)}) from e
    return result
