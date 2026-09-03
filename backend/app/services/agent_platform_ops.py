"""对话 Agent 平台能力：数据集创建、预标注、模型库、取消 Job、设备探测。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.task_types import TASK_TYPE_LABELS, normalize_task_type
from app.models.dataset import Dataset
from app.models.train import Job, ModelRecord, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner, request_cancel
from app.services import dataset_prelabel, dataset_storage, model_artifacts
from app.runners import job_control
from app.services.agent_train_ops import get_dataset, get_owned_task
from app.services.bootstrap import username_by_id
from app.services.runtime_settings import is_demo_mode


def _wizard_path(task_type: str) -> str:
    tt = normalize_task_type(task_type)
    if tt == "segment":
        return "/app/segment/wizard"
    if tt == "pose":
        return "/app/pose/wizard"
    return "/app/detect/wizard"


def _nav(label: str, path: str, query: dict[str, str] | None = None) -> dict[str, Any]:
    return {"type": "navigate", "label": label, "path": path, "query": query or {}}


def _upload_action(label: str, dataset_id: int) -> dict[str, Any]:
    """前端在对话内打开文件选择并上传到指定数据集。"""
    return {"type": "upload", "label": label, "dataset_id": int(dataset_id)}


def _annotate_action(label: str, dataset_id: int, task_type: str) -> dict[str, Any]:
    """前端在对话内打开标注抽屉（不跳转向导）。"""
    return {
        "type": "annotate",
        "label": label,
        "dataset_id": int(dataset_id),
        "task_type": normalize_task_type(task_type),
    }


def create_dataset(
    db: Session,
    user: User,
    *,
    name: str,
    task_type: str = "detect",
) -> dict[str, Any]:
    """创建空数据集，并返回上传/标注引导动作。"""
    ds_name = dataset_storage.validate_dataset_name(name)
    tt = normalize_task_type(task_type)
    conflict = (
        db.query(Dataset)
        .filter(Dataset.name == ds_name, Dataset.task_type == tt, Dataset.owner_id == user.id)
        .first()
    )
    if conflict:
        return {
            "created": False,
            "exists": True,
            "dataset_id": conflict.id,
            "name": conflict.name,
            "task_type": conflict.task_type,
            "image_count": int(conflict.image_count or 0),
            "message": f"已存在同名数据集「{conflict.name}」(#{conflict.id})。可直接在对话框点「+」或下方按钮上传图片。",
            "actions": [
                _upload_action("在对话中上传图片", conflict.id),
                _annotate_action("去标注", conflict.id, tt),
            ],
        }

    root = dataset_storage.ensure_dataset_dirs(user.username, ds_name, owner_id=user.id, task_type=tt)
    if tt == "pose":
        from app.services.pose_skeleton import coco17_config, normalize_pose_config

        meta = dataset_storage.read_meta(root)
        meta["pose"] = normalize_pose_config(coco17_config())
        meta["task_type"] = "pose"
        dataset_storage.write_meta(root, meta)

    ds = Dataset(
        name=ds_name,
        path=str(root),
        task_type=tt,
        owner_id=user.id,
        classes_json="[]",
        image_count=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {
        "created": True,
        "exists": False,
        "dataset_id": ds.id,
        "name": ds.name,
        "task_type": ds.task_type,
        "task_type_label": TASK_TYPE_LABELS.get(tt, tt),
        "image_count": 0,
        "message": "数据集已创建。请直接在对话框点「+」或下方按钮上传图片/zip，无需跳转向导。",
        "actions": [
            _upload_action("在对话中上传图片", ds.id),
            _annotate_action("去标注", ds.id, tt),
        ],
    }


def guide_dataset_upload(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """为已有数据集生成上传/标注引导（不创建）。"""
    if dataset_id is None and not (dataset_name or "").strip():
        return {
            "need_create": True,
            "message": "尚未指定数据集。可先 create_dataset，或让用户提供名称与任务类型。",
            "actions": [_nav("打开数据集管理", "/app/resources/datasets", {})],
        }
    ds = get_dataset(
        db,
        user,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
    )
    tt = ds.task_type or "detect"
    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "task_type": tt,
        "image_count": int(ds.image_count or 0),
        "message": "可直接在对话框上传图片/zip；标注请在对话页右侧面板完成。",
        "actions": [
            _upload_action("在对话中上传图片", ds.id),
            _annotate_action("去标注", ds.id, tt),
        ],
    }


def open_annotate(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    ds = get_dataset(
        db,
        user,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
    )
    tt = ds.task_type or "detect"
    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "task_type": tt,
        "message": "请在对话页右侧标注面板完成标注；打开后顶部会显示本轮任务提示。",
        "actions": [
            {
                "type": "annotate",
                "label": "去标注",
                "dataset_id": int(ds.id),
                "task_type": tt,
            },
        ],
    }


def _ensure_task_for_dataset(db: Session, ds: Dataset, user: User) -> TrainTask:
    existing = (
        db.query(TrainTask)
        .filter(TrainTask.owner_id == user.id, TrainTask.dataset_id == ds.id)
        .order_by(TrainTask.id.desc())
        .first()
    )
    if existing:
        return existing
    name = f"{ds.name}_agent"
    base = name
    n = 1
    while db.query(TrainTask).filter(TrainTask.owner_id == user.id, TrainTask.name == name).first():
        n += 1
        name = f"{base}_{n}"
    task = TrainTask(
        name=name,
        owner_id=user.id,
        dataset_id=ds.id,
        status="draft",
        step=2,
        config_json="{}",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _pick_prelabel_device() -> str:
    try:
        import torch

        if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


async def start_prelabel(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    ds = get_dataset(
        db,
        user,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
    )
    root = Path(ds.path)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes if isinstance(classes, list) else []
    dataset_storage.write_meta(root, meta)

    info = dataset_prelabel.analyze_prelabel(root, ds.task_type or "detect")
    reason = dataset_prelabel.prelabel_block_reason(info)
    if reason:
        raise ValueError(reason)

    task = _ensure_task_for_dataset(db, ds, user)
    running = (
        db.query(Job)
        .filter(
            Job.task_id == task.id,
            Job.type == "prelabel",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if running:
        return {
            "task_id": task.id,
            "dataset_id": ds.id,
            "job_id": running.id,
            "job_type": "prelabel",
            "status": running.status,
            "progress": float(running.progress or 0),
            "message": running.message or "预标注进行中",
            "wait_for_job": True,
            "already_running": True,
        }

    if not is_demo_mode():
        try:
            dataset_prelabel.resolve_prelabel_weight(ds.task_type or "detect")
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise ValueError("未安装 ultralytics，请先 pip install ultralytics") from e

    job = Job(task_id=task.id, type="prelabel", status="pending", progress=0, message="Agent：排队预标注")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit(
        "prelabel",
        {
            "job_id": job.id,
            "task_id": task.id,
            "dataset_id": ds.id,
            "device": _pick_prelabel_device(),
            "owner": username_by_id(db, user.id),
        },
    )
    db.refresh(job)
    return {
        "task_id": task.id,
        "dataset_id": ds.id,
        "job_id": job.id,
        "job_type": "prelabel",
        "status": job.status,
        "progress": float(job.progress or 0),
        "message": job.message or "已排队",
        "wait_for_job": True,
        "actions": [
            _annotate_action(
                "预标注后去抽检",
                int(ds.id),
                ds.task_type or "detect",
            ),
        ],
    }


def list_models(
    db: Session,
    user: User,
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    q = db.query(ModelRecord)
    if user.role != "admin":
        q = q.filter(ModelRecord.owner_id == user.id)
    if task_type:
        q = q.filter(ModelRecord.task_type == normalize_task_type(task_type))
    rows = q.order_by(ModelRecord.id.desc()).limit(40).all()
    items = [
        {
            "id": r.id,
            "name": r.name,
            "task_type": r.task_type,
            "task_id": r.task_id,
        }
        for r in rows
    ]
    return {
        "count": len(items),
        "models": items,
        "actions": [_nav("打开模型库", "/app/resources/models", {"type": task_type or "all"})],
    }


async def export_model_onnx(
    db: Session,
    user: User,
    *,
    model_id: int | None = None,
    task_id: int | None = None,
) -> dict[str, Any]:
    row: ModelRecord | None = None
    if model_id is not None:
        row = db.query(ModelRecord).filter(ModelRecord.id == int(model_id)).first()
        if not row:
            raise ValueError(f"模型 #{model_id} 不存在")
        if user.role != "admin" and row.owner_id != user.id:
            raise ValueError("无权使用该模型")
    elif task_id is not None:
        task = get_owned_task(db, user, int(task_id))
        row = (
            db.query(ModelRecord)
            .filter(ModelRecord.task_id == task.id)
            .order_by(ModelRecord.id.desc())
            .first()
        )
        if not row:
            raise ValueError("该训练任务尚无模型库记录，请先完成训练归档")
    else:
        raise ValueError("请提供 model_id 或 task_id")

    existing = model_artifacts.find_onnx_path(db, row)
    if existing:
        return {
            "model_id": row.id,
            "model_name": row.name,
            "already_exists": True,
            "onnx_name": existing.name,
            "message": "ONNX 已存在，可直接下载",
            "actions": [_nav("打开模型库", "/app/resources/models", {})],
        }

    pt = model_artifacts.find_pt_path(row)
    if not pt:
        raise ValueError("找不到 PT 权重，无法转 ONNX")
    if not row.task_id:
        raise ValueError("该模型缺少关联训练任务，无法自动转 ONNX")
    task = db.query(TrainTask).filter(TrainTask.id == row.task_id).first()
    if not task:
        raise ValueError("关联训练任务不存在")

    running = (
        db.query(Job)
        .filter(Job.task_id == task.id, Job.type == "export", Job.status.in_(["pending", "running"]))
        .order_by(Job.id.desc())
        .first()
    )
    if running:
        return {
            "model_id": row.id,
            "job_id": running.id,
            "job_type": "export",
            "status": running.status,
            "progress": float(running.progress or 0),
            "wait_for_job": True,
            "already_running": True,
            "message": running.message or "导出进行中",
        }

    job = Job(task_id=task.id, type="export", status="pending", progress=0, message="Agent：排队转 ONNX")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit(
        "export",
        {
            "job_id": job.id,
            "task_id": task.id,
            "formats": ["onnx"],
            "model_path": str(pt),
            "run_key": row.name,
            "model_id": row.id,
            "side_by_side": True,
        },
    )
    db.refresh(job)
    return {
        "model_id": row.id,
        "model_name": row.name,
        "job_id": job.id,
        "job_type": "export",
        "status": job.status,
        "progress": float(job.progress or 0),
        "wait_for_job": True,
        "message": job.message or "已排队转 ONNX",
    }


def cancel_job(db: Session, user: User, *, job_id: int) -> dict[str, Any]:
    job = db.query(Job).filter(Job.id == int(job_id)).first()
    if not job:
        raise ValueError(f"Job #{job_id} 不存在")
    task = db.query(TrainTask).filter(TrainTask.id == job.task_id).first()
    if task and user.role != "admin" and task.owner_id != user.id:
        raise ValueError("无权取消该 Job")
    if job.status not in ("pending", "running"):
        return {"job_id": job.id, "status": job.status, "message": "任务已结束，无需停止", "cancelled": False}
    request_cancel(job.id)
    if not job_control.is_job_active(job.id):
        job.status = "cancelled"
        job.message = "已停止"
        db.commit()
        return {"job_id": job.id, "status": "cancelled", "message": "已停止", "cancelled": True}
    if job.status == "running":
        job.message = "正在停止：将在当前 epoch 结束后停止，请稍候…"
        db.commit()
    return {
        "job_id": job.id,
        "status": job.status,
        "message": "已请求停止，将在当前 epoch 结束后停止",
        "cancelled": True,
    }


def list_devices() -> dict[str, Any]:
    devices: list[dict[str, str]] = [{"id": "cpu", "label": "CPU"}]
    cuda_available = False
    gpu_count = 0
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            gpu_count = int(torch.cuda.device_count())
            for i in range(gpu_count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:  # noqa: BLE001
                    name = f"GPU {i}"
                devices.append({"id": f"cuda:{i}", "label": f"cuda:{i} · {name}"})
    except ImportError:
        pass
    return {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "devices": devices,
        "advice": "有 GPU 时训练可选 cuda:0；否则使用 cpu（较慢）",
    }
