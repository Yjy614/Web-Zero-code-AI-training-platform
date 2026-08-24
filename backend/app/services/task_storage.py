"""训练产物与资源路径（按用户名隔离）。"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import storage_root_path, pretrained_root_path
from app.core.task_types import TASK_TYPES, normalize_task_type
from app.services.dataset_storage import owner_storage_key

SAFE_TASK_RE = re.compile(r"^[\w\u4e00-\u9fff\-]+$")

# 兼容旧引用：预训练权重按任务类型分目录
WEIGHT_TASK_TYPES = TASK_TYPES


def validate_task_name(name: str) -> str:
    """校验训练任务名称。"""
    name = name.strip()
    if not name or not SAFE_TASK_RE.match(name):
        raise ValueError("任务名称仅允许中文、字母、数字、下划线与短横线")
    return name


def normalize_weight_task_type(task_type: str | None) -> str:
    """校验并规范化权重任务类型。"""
    return normalize_task_type(task_type)

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
    """模型库用户根目录：models/<task_type>/<username>/"""
    tt = normalize_weight_task_type(task_type)
    return storage_root_path() / "models" / tt / owner_storage_key(username)


def saved_models_task_dir(username: str, task_name: str, task_type: str = "detect") -> Path:
    """某训练任务下的模型归档父目录：models/<tt>/<user>/<task_name>/"""
    return saved_models_dir(username, task_type) / Path(task_name).name


def versioned_model_dir(
    username: str, task_name: str, run_key: str, task_type: str = "detect"
) -> Path:
    """
    单次训练模型归档目录（与 runs/exports/reports 版本键一致）：
    models/<tt>/<user>/<task_name>/<run_key>/
    """
    key = resolve_run_key(task_name=task_name, model_name=run_key)
    return saved_models_task_dir(username, task_name, task_type) / key


def collect_archived_model_stems(username: str, task_name: str, task_type: str = "detect") -> set[str]:
    """收集已占用模型名（兼容旧版扁平文件 + 新版版本子目录）。"""
    stems: set[str] = set()
    root = saved_models_dir(username, task_type)
    if not root.is_dir():
        return stems
    # 旧版：直接落在用户根目录的 *.pt
    for p in root.glob("*.pt"):
        if p.is_file():
            stems.add(p.stem)
    # 新版：<task_name>/<run_key>/*.pt 或目录名即 run_key
    task_dir = saved_models_task_dir(username, task_name, task_type)
    if task_dir.is_dir():
        for child in task_dir.iterdir():
            if child.is_dir():
                stems.add(child.name)
                for p in child.glob("*.pt"):
                    if p.is_file():
                        stems.add(p.stem)
            elif child.is_file() and child.suffix.lower() in {".pt", ".pth"}:
                stems.add(child.stem)
    return stems


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


def resolve_run_key(*, task_name: str, model_path: str | None = None, model_name: str | None = None) -> str:
    """
    解析本轮训练版本键，用于 runs/exports/reports/models 下的独立子目录。
    优先 model_name；其次从 model_path 推断（…/<key>/weights/best.pt、…/<key>/<key>.pt 或 …/<key>.pt）。
    """
    if model_name and str(model_name).strip():
        return Path(str(model_name).strip()).name
    mp = (model_path or "").strip()
    if mp:
        p = Path(mp)
        if p.parent.name.lower() == "weights":
            key = p.parent.parent.name
            if key:
                return key
        # 新版模型归档：…/<task>/<run_key>/<run_key>.pt
        if p.stem and p.parent.name == p.stem:
            return p.stem
        stem = p.stem
        if stem:
            return stem
    base = (task_name or "task").strip() or "task"
    return Path(base).name


def versioned_exports_dir(
    username: str, task_name: str, run_key: str, task_type: str = "detect"
) -> Path:
    """单次训练的导出目录：exports/<tt>/<user>/<task>/<run_key>/"""
    return exports_dir(username, task_name, task_type) / resolve_run_key(
        task_name=task_name, model_name=run_key
    )


def versioned_reports_dir(
    username: str, task_name: str, run_key: str, task_type: str = "detect"
) -> Path:
    """单次训练的报告目录：reports/<tt>/<user>/<task>/<run_key>/"""
    return reports_dir(username, task_name, task_type) / resolve_run_key(
        task_name=task_name, model_name=run_key
    )


def safe_join_under(root: Path, relative: str) -> Path:
    """将相对路径安全拼到 root 下，禁止 .. 逃逸。"""
    rel = Path(str(relative).replace("\\", "/"))
    if rel.is_absolute() or not rel.parts or any(p in {"", ".", ".."} for p in rel.parts):
        raise ValueError("非法相对路径")
    candidate = root.joinpath(*rel.parts).resolve()
    root_res = root.resolve()
    if candidate != root_res and root_res not in candidate.parents:
        raise ValueError("路径越界")
    return candidate


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
