"""训练产物与资源路径（按用户名隔离）。"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import storage_root_path, pretrained_root_path
from app.services.dataset_storage import owner_storage_key

SAFE_TASK_RE = re.compile(r"^[\w\u4e00-\u9fff\-]+$")

# 预训练权重按任务类型分目录：pretrained/<task_type>/（系统级，不按用户）
WEIGHT_TASK_TYPES = ("detect", "segment")


def validate_task_name(name: str) -> str:
    """校验训练任务名称。"""
    name = name.strip()
    if not name or not SAFE_TASK_RE.match(name):
        raise ValueError("任务名称仅允许中文、字母、数字、下划线与短横线")
    return name


def normalize_weight_task_type(task_type: str | None) -> str:
    """校验并规范化权重任务类型。"""
    t = (task_type or "detect").strip().lower()
    if t not in WEIGHT_TASK_TYPES:
        raise ValueError(f"不支持的任务类型：{task_type}，可选：{', '.join(WEIGHT_TASK_TYPES)}")
    return t


def runs_dir(username: str, task_name: str, task_type: str = "detect") -> Path:
    """训练输出目录：runs/<task_type>/<username>/<task>/"""
    tt = normalize_weight_task_type(task_type)
    return storage_root_path() / "runs" / tt / owner_storage_key(username) / task_name


def exports_dir(username: str, task_name: str, task_type: str = "detect") -> Path:
    """导出目录：exports/<task_type>/<username>/<task>/"""
    tt = normalize_weight_task_type(task_type)
    return storage_root_path() / "exports" / tt / owner_storage_key(username) / task_name


def reports_dir(username: str, task_name: str, task_type: str = "detect") -> Path:
    """报告目录：reports/<task_type>/<username>/<task>/"""
    tt = normalize_weight_task_type(task_type)
    return storage_root_path() / "reports" / tt / owner_storage_key(username) / task_name


def saved_models_dir(username: str, task_type: str = "detect") -> Path:
    """模型库归档目录（每次训练单独副本，避免同数据集互相覆盖）。"""
    tt = normalize_weight_task_type(task_type)
    return storage_root_path() / "models" / tt / owner_storage_key(username)


def weights_dir(task_type: str = "detect") -> Path:
    """系统预训练权重目录（管理员上传，按任务类型隔离，不按用户拆）。"""
    tt = normalize_weight_task_type(task_type)
    return pretrained_root_path() / tt


def ensure_task_dirs(username: str, task_name: str, task_type: str = "detect") -> dict[str, Path]:
    """创建 runs / exports / reports 子目录。"""
    paths = {
        "runs": runs_dir(username, task_name, task_type),
        "exports": exports_dir(username, task_name, task_type),
        "reports": reports_dir(username, task_name, task_type),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    (paths["runs"] / "weights").mkdir(parents=True, exist_ok=True)
    return paths


def list_weight_files(task_type: str = "detect") -> list[dict]:
    """列出指定任务类型下的预训练权重文件（不下载）。"""
    tt = normalize_weight_task_type(task_type)
    root = weights_dir(tt)
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".onnx"}:
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    "task_type": tt,
                }
            )
    return items


def delete_weight_file(task_type: str, name: str) -> None:
    """删除指定任务类型下的权重文件。"""
    tt = normalize_weight_task_type(task_type)
    fname = Path(name).name
    if not fname or fname in {".", ".."}:
        raise ValueError("非法文件名")
    path = weights_dir(tt) / fname
    if not path.is_file():
        raise FileNotFoundError(f"权重不存在：{fname}")
    path.unlink()
