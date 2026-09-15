"""模型溯源信息：登记训练产出时写入，供主动学习/续训等下游使用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.train import TrainTask
from app.services import dataset_storage


def parse_parent_model_id(pretrained_weight: str | None) -> int | None:
    """从 @model:{id} 解析父模型 ID。"""
    raw = (pretrained_weight or "").strip()
    if not raw.startswith("@model:"):
        return None
    try:
        return int(raw.split(":", 1)[1])
    except ValueError:
        return None


def build_model_lineage(db: Session, task: TrainTask, *, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    组装写入 ModelRecord.metrics_json 的 lineage 段。
    不替代原有 map50/loss 等指标字段。
    """
    cfg: dict[str, Any] = {}
    try:
        raw = json.loads(task.config_json or "{}")
        if isinstance(raw, dict):
            cfg = raw
    except json.JSONDecodeError:
        cfg = {}

    weight = str(cfg.get("pretrained_weight") or "")
    parent_model_id = parse_parent_model_id(weight)
    # 仅模型库续训（@model:{id}）记为父权重；仓库里的官方预训练不算「续训」
    parent_label = _display_parent_weight(db, parent_model_id, weight) if parent_model_id else None

    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first() if task.dataset_id else None
    classes: list[str] = []
    image_count = 0
    dataset_name = None
    dataset_id = int(task.dataset_id) if task.dataset_id else None
    if ds:
        dataset_name = ds.name
        dataset_id = int(ds.id)
        try:
            loaded = json.loads(ds.classes_json or "[]")
            if isinstance(loaded, list):
                classes = [str(c) for c in loaded]
        except json.JSONDecodeError:
            classes = []
        if ds.path:
            try:
                image_count = dataset_storage.count_active_images(Path(ds.path))
            except Exception:  # noqa: BLE001
                image_count = int(ds.image_count or 0)

    train_config = {
        "epochs": cfg.get("epochs"),
        "batch": cfg.get("batch"),
        "device": cfg.get("device"),
        "imgsz": cfg.get("imgsz"),
        "pretrained_weight": weight or None,
        "augment": cfg.get("augment"),
        "train_strategy": cfg.get("train_strategy"),
    }

    task_type = (ds.task_type if ds else None) or "detect"
    lineage: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task_id": int(task.id),
        "task_name": task.name,
        "task_type": task_type,
        "train_config": train_config,
        "parent_weight": weight or None,
        "parent_model_id": parent_model_id,
        "parent_weight_label": parent_label,
        "image_count": image_count,
        "class_count": len(classes),
        "classes": classes,
    }
    # 姿态：额外留存骨架配置，供主动学习临时集 / 续训对齐
    if task_type == "pose" and ds and ds.path:
        try:
            meta = dataset_storage.read_meta(Path(ds.path))
            pose_cfg = meta.get("pose")
            if isinstance(pose_cfg, dict):
                lineage["pose"] = pose_cfg
        except Exception:  # noqa: BLE001
            pass

    # 合并调用方传入的指标，避免覆盖 lineage
    out: dict[str, Any] = dict(metrics or {})
    out["lineage"] = lineage
    return out


def _display_parent_weight(db: Session, parent_model_id: int | None, parent_weight: str | None) -> str | None:
    """续训展示名：优先父模型归档 PT 文件名，例如 ties.pt。"""
    from app.models.train import ModelRecord
    from app.services import model_artifacts

    if parent_model_id is None:
        return None
    prow = db.query(ModelRecord).filter(ModelRecord.id == int(parent_model_id)).first()
    if prow:
        pt = model_artifacts.find_pt_path(prow)
        if pt is not None:
            return pt.name
        stem = str(prow.name or "").strip()
        if stem:
            return stem if stem.lower().endswith(".pt") else f"{stem}.pt"
    raw = str(parent_weight or "").strip()
    if raw and not raw.startswith("@model:"):
        return Path(raw).name
    return None


def parent_weight_label(db: Session, lineage: dict[str, Any]) -> tuple[int | None, str | None, str | None]:
    """
    解析续训来源：父模型 id、原始权重字段、展示用权重文件名。
    仅当从模型库 @model:{id} 续训时返回展示名；官方预训练不显示「续训自」。
    """
    if not isinstance(lineage, dict):
        return None, None, None
    parent_weight = str(lineage.get("parent_weight") or "").strip() or None
    raw_id = lineage.get("parent_model_id")
    parent_model_id: int | None = None
    if raw_id is not None:
        try:
            parent_model_id = int(raw_id)
        except (TypeError, ValueError):
            parent_model_id = None
    if parent_model_id is None:
        parent_model_id = parse_parent_model_id(parent_weight)

    stored = str(lineage.get("parent_weight_label") or "").strip() or None
    display = stored if parent_model_id is not None else None
    if parent_model_id is not None and not display:
        display = _display_parent_weight(db, parent_model_id, parent_weight)
    return parent_model_id, parent_weight, display
