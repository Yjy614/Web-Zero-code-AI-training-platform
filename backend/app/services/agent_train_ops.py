"""对话 Agent 的训练相关操作（创建任务、写配置、开训/评估、查 Job、抽样推理）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.task_types import normalize_task_type
from app.models.dataset import Dataset
from app.models.train import Job, ModelRecord, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner
from app.schemas.task import TrainConfig
from app.services import dataset_prelabel, model_artifacts, model_infer, task_storage, yolo_split
from app.services.agent_orchestrate import _default_weight_name, _normalize_train_device
from app.services.runtime_settings import is_demo_mode
from app.services.weight_resolve import normalize_weight_name, resolve_pretrained_weight

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_dataset(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> Dataset:
    if dataset_id is not None:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            raise ValueError(f"数据集 id={dataset_id} 不存在")
        if user.role != "admin" and ds.owner_id != user.id:
            raise ValueError("无权访问该数据集")
        return ds
    name = (dataset_name or "").strip()
    if not name:
        raise ValueError("请提供 dataset_id 或 dataset_name")
    tt = normalize_task_type(task_type or "detect")
    q = db.query(Dataset).filter(Dataset.name == name, Dataset.task_type == tt)
    if user.role != "admin":
        q = q.filter(Dataset.owner_id == user.id)
    ds = q.first()
    if not ds:
        raise ValueError(f"未找到数据集「{name}」（类型 {tt}）")
    return ds


def get_owned_task(db: Session, user: User, task_id: int) -> TrainTask:
    task = db.query(TrainTask).filter(TrainTask.id == int(task_id)).first()
    if not task:
        raise ValueError(f"训练任务 #{task_id} 不存在")
    if user.role != "admin" and task.owner_id != user.id:
        raise ValueError("无权使用该训练任务")
    return task


def ensure_train_task(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
    task_name: str | None = None,
    task_id: int | None = None,
) -> dict[str, Any]:
    """确保有一个绑定到数据集的训练任务。"""
    if task_id is not None:
        task = get_owned_task(db, user, task_id)
        ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
        return {
            "task_id": task.id,
            "task_name": task.name,
            "dataset_id": task.dataset_id,
            "dataset_name": ds.name if ds else None,
            "task_type": ds.task_type if ds else None,
            "status": task.status,
            "reused": True,
        }

    ds = get_dataset(
        db,
        user,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
    )
    name = (task_name or f"{ds.name}_agent").strip() or f"{ds.name}_agent"
    base = name
    n = 1
    while db.query(TrainTask).filter(TrainTask.owner_id == user.id, TrainTask.name == name).first():
        n += 1
        name = f"{base}_{n}"

    # 尽量复用同数据集、同名前缀的最近草稿任务，避免每次新建
    existing = (
        db.query(TrainTask)
        .filter(TrainTask.owner_id == user.id, TrainTask.dataset_id == ds.id)
        .order_by(TrainTask.id.desc())
        .first()
    )
    if existing and existing.status in {"draft", "configured", "trained", "evaluated", "exported"}:
        return {
            "task_id": existing.id,
            "task_name": existing.name,
            "dataset_id": ds.id,
            "dataset_name": ds.name,
            "task_type": ds.task_type,
            "status": existing.status,
            "reused": True,
        }

    task = TrainTask(
        name=name,
        owner_id=user.id,
        dataset_id=ds.id,
        status="draft",
        step=3,
        config_json="{}",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    try:
        task_storage.ensure_task_dirs(user.username, task.name, ds.task_type or "detect")
    except Exception:  # noqa: BLE001
        pass
    return {
        "task_id": task.id,
        "task_name": task.name,
        "dataset_id": ds.id,
        "dataset_name": ds.name,
        "task_type": ds.task_type,
        "status": task.status,
        "reused": False,
    }


def set_train_config(
    db: Session,
    user: User,
    *,
    task_id: int,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    pretrained_weight: str | None = None,
    imgsz: int | None = None,
) -> dict[str, Any]:
    task = get_owned_task(db, user, task_id)
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds:
        raise ValueError("任务绑定的数据集不存在")
    tt = ds.task_type or "detect"

    raw: dict[str, Any] = {}
    try:
        raw = json.loads(task.config_json or "{}")
        if not isinstance(raw, dict):
            raw = {}
    except json.JSONDecodeError:
        raw = {}

    if epochs is not None:
        raw["epochs"] = int(epochs)
    if batch is not None:
        raw["batch"] = int(batch)
    if imgsz is not None:
        raw["imgsz"] = int(imgsz)
    if device is not None:
        raw["device"] = _normalize_train_device(str(device))
    else:
        raw["device"] = _normalize_train_device(str(raw.get("device") or "cpu"))

    weight = normalize_weight_name(str(pretrained_weight if pretrained_weight is not None else raw.get("pretrained_weight") or ""))
    if not weight:
        weight = _default_weight_name(tt)
    raw["pretrained_weight"] = weight

    try:
        resolve_pretrained_weight(weight, tt)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e

    cfg = TrainConfig.model_validate(raw)
    task.config_json = cfg.model_dump_json()
    task.status = "configured"
    task.step = max(task.step or 0, 3)
    db.commit()
    return {"task_id": task.id, "config": cfg.model_dump(), "task_type": tt}


def _assert_can_train(ds: Dataset) -> dict[str, Any]:
    info = dataset_prelabel.analyze_prelabel(Path(ds.path), ds.task_type or "detect")
    labeled = int(info.get("labeled_count") or 0)
    class_count = int(info.get("class_count") or 0)
    if class_count < 1:
        raise ValueError("请先添加至少一个类别后再训练")
    if labeled < 1:
        raise ValueError("尚无已标注图片，请先到对应向导完成标注后再训练")
    return info


async def start_train(db: Session, user: User, *, task_id: int) -> dict[str, Any]:
    task = get_owned_task(db, user, task_id)
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds:
        raise ValueError("任务绑定的数据集不存在")

    label_info = _assert_can_train(ds)
    raw = {}
    try:
        raw = json.loads(task.config_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    if not raw.get("pretrained_weight"):
        set_train_config(db, user, task_id=task.id)
        db.refresh(task)
        raw = json.loads(task.config_json or "{}")

    cfg = TrainConfig.model_validate(raw)
    root = Path(ds.path)
    if not (root / "data.yaml").exists():
        yolo_split.prepare_yolo_split(
            root,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )

    running = (
        db.query(Job)
        .filter(Job.task_id == task.id, Job.type == "train", Job.status.in_(["pending", "running"]))
        .first()
    )
    if running:
        return {
            "task_id": task.id,
            "job_id": running.id,
            "job_type": "train",
            "status": running.status,
            "progress": float(running.progress or 0),
            "message": running.message or "训练进行中",
            "wait_for_job": True,
            "already_running": True,
            "labeled_count": label_info.get("labeled_count"),
        }

    if not is_demo_mode():
        try:
            resolve_pretrained_weight(cfg.pretrained_weight, ds.task_type or "detect")
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise ValueError("未安装 ultralytics，请在后端执行：pip install ultralytics") from e

    job = Job(task_id=task.id, type="train", status="pending", progress=0, message="Agent：排队训练")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit("train", {"job_id": job.id, "epochs": cfg.epochs, "task_id": task.id})
    db.refresh(job)
    return {
        "task_id": task.id,
        "job_id": job.id,
        "job_type": "train",
        "status": job.status,
        "progress": float(job.progress or 0),
        "message": job.message or "已排队",
        "wait_for_job": True,
        "epochs": cfg.epochs,
        "device": cfg.device,
        "pretrained_weight": cfg.pretrained_weight,
        "labeled_count": label_info.get("labeled_count"),
    }


async def start_eval(db: Session, user: User, *, task_id: int) -> dict[str, Any]:
    task = get_owned_task(db, user, task_id)
    if task.status not in {"trained", "evaluated", "exported"} and not task.model_path:
        raise ValueError("请先完成训练再评估")
    running = (
        db.query(Job)
        .filter(Job.task_id == task.id, Job.type == "eval", Job.status.in_(["pending", "running"]))
        .first()
    )
    if running:
        return {
            "task_id": task.id,
            "job_id": running.id,
            "job_type": "eval",
            "status": running.status,
            "progress": float(running.progress or 0),
            "message": running.message or "评估进行中",
            "wait_for_job": True,
            "already_running": True,
        }
    job = Job(task_id=task.id, type="eval", status="pending", progress=0, message="Agent：排队评估")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit("eval", {"job_id": job.id, "task_id": task.id})
    db.refresh(job)
    return {
        "task_id": task.id,
        "job_id": job.id,
        "job_type": "eval",
        "status": job.status,
        "progress": float(job.progress or 0),
        "message": job.message or "已排队",
        "wait_for_job": True,
    }


def get_job_status(db: Session, user: User, *, job_id: int) -> dict[str, Any]:
    job = db.query(Job).filter(Job.id == int(job_id)).first()
    if not job:
        raise ValueError(f"Job #{job_id} 不存在")
    task = db.query(TrainTask).filter(TrainTask.id == job.task_id).first()
    if task and user.role != "admin" and task.owner_id != user.id:
        raise ValueError("无权查看该 Job")
    result: dict[str, Any] = {}
    try:
        result = json.loads(job.result_json or "{}")
        if not isinstance(result, dict):
            result = {}
    except json.JSONDecodeError:
        result = {}
    metrics: dict[str, Any] = {}
    if task and task.metrics_json:
        try:
            metrics = json.loads(task.metrics_json)
            if not isinstance(metrics, dict):
                metrics = {}
        except json.JSONDecodeError:
            metrics = {}
    done = job.status in {"completed", "failed", "cancelled"}
    return {
        "job_id": job.id,
        "task_id": job.task_id,
        "job_type": job.type,
        "status": job.status,
        "progress": float(job.progress or 0),
        "message": job.message or "",
        "done": done,
        "result_keys": list(result.keys())[:20],
        "task_status": task.status if task else None,
        "model_path": (task.model_path if task else "") or "",
        "metrics_summary": _metrics_summary(metrics),
    }


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("map50", "map50_95", "precision", "recall", "fitness", "box_loss", "run_key")
    out: dict[str, Any] = {}
    for k in keys:
        if k in metrics and metrics[k] is not None:
            out[k] = metrics[k]
    # 常见嵌套
    for nest in ("metrics", "summary"):
        sub = metrics.get(nest)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and k not in out:
                    out[k] = sub[k]
    return out


def _pick_sample_image(dataset_root: Path) -> Path | None:
    for sub in ("images/val", "images/test", "images/train", "images", "raw"):
        d = dataset_root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return p
    return None


def run_sample_infer(
    db: Session,
    user: User,
    *,
    task_id: int | None = None,
    model_id: int | None = None,
) -> dict[str, Any]:
    """用任务产物或模型库权重，对数据集中一张样例图做推理验证。"""
    row: ModelRecord | None = None
    task: TrainTask | None = None
    tt = "detect"
    ds: Dataset | None = None

    if model_id is not None:
        row = db.query(ModelRecord).filter(ModelRecord.id == int(model_id)).first()
        if not row:
            raise ValueError(f"模型 #{model_id} 不存在")
        if user.role != "admin" and row.owner_id != user.id:
            raise ValueError("无权使用该模型")
        tt = row.task_type or "detect"
        if row.task_id:
            task = db.query(TrainTask).filter(TrainTask.id == row.task_id).first()
    elif task_id is not None:
        task = get_owned_task(db, user, task_id)
        ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
        tt = (ds.task_type if ds else None) or "detect"
        row = (
            db.query(ModelRecord)
            .filter(ModelRecord.task_id == task.id)
            .order_by(ModelRecord.id.desc())
            .first()
        )
        if not row and task.model_path:
            # 直接用 task.model_path
            weight = Path(task.model_path)
            if not weight.is_file():
                raise ValueError("训练产物权重文件不存在，请先完成训练或导出到模型库")
            if not ds:
                raise ValueError("找不到数据集，无法抽样推理")
            img = _pick_sample_image(Path(ds.path))
            if not img:
                raise ValueError("数据集中没有可用图片")
            content = img.read_bytes()
            result = model_infer.run_predict(
                weight,
                content,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                task_type=tt,
                model_format="pt",
            )
            return _infer_summary(result, image_name=img.name, source="task_model_path", task_id=task.id)
    else:
        raise ValueError("请提供 task_id 或 model_id")

    if not row:
        raise ValueError("未找到可推理的模型记录，请先完成训练并归档到模型库")

    if not ds and task:
        ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds:
        # 尝试按 owner 找任意同类型数据集
        q = db.query(Dataset).filter(Dataset.task_type == tt)
        if user.role != "admin":
            q = q.filter(Dataset.owner_id == user.id)
        ds = q.order_by(Dataset.id.desc()).first()
    if not ds:
        raise ValueError("找不到可用于抽样的数据集")

    pt = model_artifacts.find_pt_path(row)
    onnx = model_artifacts.find_onnx_path(db, row)
    weight, fmt = model_infer.resolve_infer_weight(pt=pt, onnx=onnx)
    img = _pick_sample_image(Path(ds.path))
    if not img:
        raise ValueError("数据集中没有可用图片")
    content = img.read_bytes()
    result = model_infer.run_predict(
        weight,
        content,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        task_type=tt,
        model_format=fmt,
    )
    return _infer_summary(
        result,
        image_name=img.name,
        source="model_record",
        task_id=task.id if task else None,
        model_id=row.id,
        model_name=row.name,
    )


def _infer_summary(
    result: dict[str, Any],
    *,
    image_name: str,
    source: str,
    task_id: int | None = None,
    model_id: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    dets = result.get("detections") if isinstance(result.get("detections"), list) else []
    classes: dict[str, int] = {}
    for d in dets:
        if not isinstance(d, dict):
            continue
        name = str(d.get("label") or d.get("class_name") or d.get("cls") or "obj")
        classes[name] = classes.get(name, 0) + 1
    preview = result.get("image_base64") or result.get("preview_base64")
    return {
        "ok_infer": True,
        "source": source,
        "task_id": task_id,
        "model_id": model_id,
        "model_name": model_name,
        "sample_image": image_name,
        "detection_count": len(dets),
        "class_counts": classes,
        "usable_hint": "有检出目标，模型基本可用" if dets else "未检出目标，可能需更多标注/训练或换图验证",
        "has_preview": bool(preview),
        # 预览给前端卡片用；写入 LLM 上下文前会被剥离
        "preview_base64": preview if isinstance(preview, str) else None,
    }
