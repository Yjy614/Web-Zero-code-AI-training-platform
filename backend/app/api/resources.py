"""权重仓库与模型库 API。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.core.database import get_db
from app.models.train import Job, ModelRecord, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner
from app.schemas.task import JobOut, ModelOut, PredictOut, WeightItem
from app.services import model_artifacts, model_infer, task_storage

router = APIRouter(tags=["资源"])


def _parse_task_type(task_type: str) -> str:
    try:
        return task_storage.normalize_weight_task_type(task_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_task_type", "message": str(e)}) from e


def _parse_json(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        task_id=job.task_id,
        type=job.type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=_parse_json(job.result_json),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _model_out(db: Session, r: ModelRecord) -> ModelOut:
    try:
        metrics = json.loads(r.metrics_json or "{}")
    except json.JSONDecodeError:
        metrics = {}
    return ModelOut(
        id=r.id,
        name=r.name,
        path="",
        task_type=r.task_type,
        owner_id=r.owner_id,
        task_id=r.task_id,
        metrics=metrics if isinstance(metrics, dict) else {},
        created_at=r.created_at,
        has_pt=model_artifacts.find_pt_path(r) is not None,
        has_onnx=model_artifacts.find_onnx_path(db, r) is not None,
    )


def _get_accessible_model(db: Session, model_id: int, user: User) -> ModelRecord:
    row = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not row:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "模型不存在"})
    if user.role != "admin" and row.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问该模型"})
    return row


@router.get("/weights/task-types")
def list_weight_task_types(user: User = Depends(get_current_user)) -> dict:
    """返回权重仓库支持的任务类型（便于前端扩展）。"""
    _ = user
    labels = {
        "detect": "目标检测",
        "segment": "实例分割",
        "pose": "姿态估计",
    }
    return {
        "items": [
            {
                "task_type": t,
                "label": labels.get(t, t),
                "enabled": True,
            }
            for t in task_storage.WEIGHT_TASK_TYPES
        ]
    }


@router.get("/weights", response_model=list[WeightItem])
def list_weights(task_type: str = "detect", user: User = Depends(get_current_user)) -> list[WeightItem]:
    """列出指定任务类型下的预训练权重。"""
    _ = user
    tt = _parse_task_type(task_type)
    return [
        WeightItem(name=x["name"], path="", size=x["size"], task_type=x["task_type"])
        for x in task_storage.list_weight_files(tt)
    ]


@router.post("/weights/upload", response_model=WeightItem)
async def upload_weight(
    file: UploadFile = File(...),
    task_type: str = Form("detect"),
    _: User = Depends(require_admin),
) -> WeightItem:
    """管理员上传预训练权重到 pretrained/<task_type>/。"""
    tt = _parse_task_type(task_type)
    name = Path(file.filename or "weight.pt").name
    if Path(name).suffix.lower() not in {".pt", ".pth", ".onnx"}:
        raise HTTPException(status_code=400, detail={"code": "bad_file", "message": "仅支持 .pt / .pth / .onnx"})
    root = task_storage.weights_dir(tt)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    content = await file.read()
    target.write_bytes(content)
    return WeightItem(name=target.name, path="", size=target.stat().st_size, task_type=tt)


@router.delete("/weights/{name}")
def delete_weight(
    name: str,
    task_type: str = "detect",
    _: User = Depends(require_admin),
) -> dict:
    """管理员删除指定任务类型下的权重文件。"""
    tt = _parse_task_type(task_type)
    try:
        task_storage.delete_weight_file(tt, name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(e)}) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": str(e)}) from e
    return {"message": "已删除", "name": Path(name).name, "task_type": tt}


@router.get("/models", response_model=list[ModelOut])
def list_models(
    task_type: str | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ModelOut]:
    """模型库：训练产出记录（可按 task_type 过滤）。"""
    q = db.query(ModelRecord)
    if user.role != "admin":
        q = q.filter(ModelRecord.owner_id == user.id)
    if task_type:
        tt = _parse_task_type(task_type)
        q = q.filter(ModelRecord.task_type == tt)
    rows = q.order_by(ModelRecord.id.desc()).all()
    return [_model_out(db, r) for r in rows]


@router.get("/models/{model_id}/download")
def download_model(
    model_id: int,
    format: str = Query("pt", description="pt 或 onnx"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """下载训练产出的 PT / ONNX 模型文件。"""
    row = _get_accessible_model(db, model_id, user)
    fmt = (format or "pt").strip().lower()
    if fmt == "pt":
        path = model_artifacts.find_pt_path(row)
        if not path:
            raise HTTPException(status_code=404, detail={"code": "file_missing", "message": "PT 模型文件不存在"})
        return FileResponse(path, filename=path.name or f"{row.name}.pt")
    if fmt == "onnx":
        path = model_artifacts.find_onnx_path(db, row)
        if not path:
            raise HTTPException(
                status_code=404,
                detail={"code": "onnx_missing", "message": "尚未生成 ONNX，请先转格式"},
            )
        return FileResponse(path, filename=path.name or f"{row.name}.onnx")
    raise HTTPException(status_code=400, detail={"code": "bad_format", "message": "format 仅支持 pt 或 onnx"})


@router.post("/models/{model_id}/export-onnx", response_model=JobOut)
async def export_model_onnx(
    model_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    """若尚无 ONNX，则启动仅导出 onnx 的任务；已有则返回最近一次已完成导出 Job 信息。"""
    row = _get_accessible_model(db, model_id, user)
    existing = model_artifacts.find_onnx_path(db, row)
    if existing:
        # 已有文件：返回一个已完成占位 Job，方便前端统一处理
        return JobOut(
            id=0,
            task_id=int(row.task_id or 0),
            type="export",
            status="completed",
            progress=100,
            message="ONNX 已存在",
            result={"files": [{"format": "onnx", "name": existing.name}]},
        )

    pt = model_artifacts.find_pt_path(row)
    if not pt:
        raise HTTPException(status_code=400, detail={"code": "no_pt", "message": "找不到 PT 权重，无法转 ONNX"})

    if not row.task_id:
        raise HTTPException(
            status_code=400,
            detail={"code": "no_task", "message": "该模型缺少关联训练任务，无法自动转 ONNX"},
        )
    task = db.query(TrainTask).filter(TrainTask.id == row.task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "关联训练任务不存在"})

    # 同任务若已有导出在跑：仍返回该 Job（前端会轮询）；真正导出用本模型 PT，见 payload
    running = (
        db.query(Job)
        .filter(Job.task_id == task.id, Job.type == "export", Job.status.in_(["pending", "running"]))
        .order_by(Job.id.desc())
        .first()
    )
    if running:
        return _job_out(running)

    job = Job(task_id=task.id, type="export", status="pending", progress=0, message="排队转 ONNX…")
    db.add(job)
    db.commit()
    db.refresh(job)
    # 必须带上本卡片对应的 PT / run_key，避免误用任务上「最新一次」训练路径
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
    return _job_out(job)


@router.post("/models/{model_id}/predict", response_model=PredictOut)
async def predict_model(
    model_id: int,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictOut:
    """
    推理试用：上传一张图片，用指定模型库权重做检测/分割，
    返回可视化 JPEG（base64）与 detections 列表。
    """
    row = _get_accessible_model(db, model_id, user)
    pt = model_artifacts.find_pt_path(row)
    onnx = model_artifacts.find_onnx_path(db, row)
    try:
        weight, fmt = model_infer.resolve_infer_weight(pt=pt, onnx=onnx)
    except model_infer.InferError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message}) from e

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"code": "empty_file", "message": "请上传图片文件"})
    # 简单大小限制（20MB）
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"code": "too_large", "message": "图片过大（上限 20MB）"})

    try:
        result = model_infer.run_predict(
            weight,
            content,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            task_type=row.task_type or "detect",
            model_format=fmt,
        )
    except model_infer.InferError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message}) from e

    return PredictOut(**result)


@router.delete("/models/{model_id}")
def delete_model(
    model_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """删除模型记录，并清理权重、训练过程图、导出与报告等关联目录。"""
    row = _get_accessible_model(db, model_id, user)
    return model_artifacts.delete_model_artifacts(db, row)
