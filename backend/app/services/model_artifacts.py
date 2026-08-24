"""模型库产物路径解析与清理。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.models.train import ModelRecord, TrainTask
from app.services import task_storage
from app.services.bootstrap import username_by_id


def resolve_task_for_model(db: Session, row: ModelRecord) -> TrainTask | None:
    if not row.task_id:
        return None
    return db.query(TrainTask).filter(TrainTask.id == row.task_id).first()


def next_unique_model_name(
    db: Session,
    owner_id: int,
    base: str,
    *,
    archive_dir: Path | None = None,
    archived_stems: set[str] | None = None,
) -> str:
    """
    生成不重复模型名：首次用 base，其后 base_2、base_3…
    会把历史名（base-best、base-时间戳）和归档目录里已有文件/子目录一并视为占用。
    """
    import re

    base = (base or "model").strip() or "model"
    existing: set[str] = set()
    for (n,) in db.query(ModelRecord.name).filter(ModelRecord.owner_id == owner_id).all():
        if n:
            existing.add(str(n))
    if archived_stems:
        existing.update(archived_stems)
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for p in archive_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".onnx"}:
                existing.add(p.stem)
            elif p.is_dir() and p.parent == archive_dir:
                existing.add(p.name)

    used_slots: set[int] = set()
    for n in existing:
        if n == base or n == f"{base}-best":
            used_slots.add(1)
            continue
        m = re.fullmatch(rf"{re.escape(base)}_(\d+)", n)
        if m:
            used_slots.add(int(m.group(1)))
            continue
        # 历史时间戳命名：name-YYYYMMDD-HHMMSS
        if re.fullmatch(rf"{re.escape(base)}-\d{{8}}-\d{{6}}", n):
            used_slots.add(1)
            continue

    if 1 not in used_slots:
        return base
    idx = 2
    while idx in used_slots or f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def artifact_dirs_for_model(db: Session, row: ModelRecord) -> tuple[TrainTask, dict[str, Path]] | None:
    """返回 (task, {runs, exports, reports})。"""
    task = resolve_task_for_model(db, row)
    if not task:
        return None
    username = username_by_id(db, task.owner_id)
    tt = row.task_type or "detect"
    return task, {
        "runs": task_storage.runs_dir(username, task.name, tt),
        "exports": task_storage.exports_dir(username, task.name, tt),
        "reports": task_storage.reports_dir(username, task.name, tt),
    }


def find_pt_path(row: ModelRecord) -> Path | None:
    path = Path(row.path) if row.path else None
    if path and path.is_file():
        return path
    return None


def find_onnx_path(db: Session, row: ModelRecord) -> Path | None:
    """
    查找本模型版本对应的 ONNX。
    仅匹配旁路文件与本 run_key 导出路径，绝不串用同任务其他版本的 onnx。
    """
    candidates: list[Path] = []
    pt = find_pt_path(row)
    if pt:
        candidates.append(pt.with_suffix(".onnx"))

    run_key = (row.name or "").strip()
    resolved = artifact_dirs_for_model(db, row)
    if resolved:
        task, dirs = resolved
        export_root = dirs["exports"]
        key = run_key or task_storage.resolve_run_key(
            task_name=task.name, model_path=row.path or task.model_path
        )
        # 新版：exports/<tt>/<user>/<task>/<run_key>/<run_key>.onnx
        versioned = export_root / key
        candidates.append(versioned / f"{key}.onnx")
        candidates.append(versioned / f"{key}_demo.onnx")
        # 旧版扁平：exports/.../<task>/<run_key>.onnx
        candidates.append(export_root / f"{key}.onnx")
        candidates.append(export_root / f"{key}_demo.onnx")

    seen: set[str] = set()
    for p in candidates:
        try:
            key_s = str(p.resolve()) if p.exists() else str(p)
        except OSError:
            key_s = str(p)
        if key_s in seen:
            continue
        seen.add(key_s)
        if p.is_file():
            return p
    return None


def delete_model_artifacts(db: Session, row: ModelRecord) -> dict:
    """
    删除单条模型记录，并清理该版本对应的：
    - models 归档 PT / 旁路 ONNX
    - runs/<run_key>/、exports/<run_key>/、reports/<run_key>/
    同任务其他版本保留。
    """
    import shutil

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    run_key = (row.name or "").strip()
    tt = row.task_type or "detect"
    resolved = artifact_dirs_for_model(db, row)

    pt = find_pt_path(row)
    # 新版：整目录删除 models/<tt>/<user>/<task>/<run_key>/
    model_dir_removed = False
    if resolved and run_key:
        task, _dirs = resolved
        username = username_by_id(db, task.owner_id)
        versioned_model = task_storage.versioned_model_dir(username, task.name, run_key, tt)
        if versioned_model.is_dir():
            try:
                shutil.rmtree(versioned_model)
                removed_dirs.append(str(versioned_model))
                model_dir_removed = True
            except OSError:
                pass
        # 若版本目录为空父级可顺手清理（失败忽略）
        task_models = task_storage.saved_models_task_dir(username, task.name, tt)
        if task_models.is_dir() and not any(task_models.iterdir()):
            try:
                task_models.rmdir()
            except OSError:
                pass

    # 旧版扁平文件，或无关联任务时按路径删除
    if not model_dir_removed and pt and pt.is_file():
        parent = pt.parent
        # 若已是版本目录结构（父目录名=模型名），整目录删除
        if run_key and parent.name == run_key:
            try:
                shutil.rmtree(parent)
                removed_dirs.append(str(parent))
                model_dir_removed = True
            except OSError:
                pass
        if not model_dir_removed:
            side = pt.with_suffix(".onnx")
            try:
                pt.unlink()
                removed_files.append(str(pt))
            except OSError:
                pass
            if side.is_file():
                try:
                    side.unlink()
                    removed_files.append(str(side))
                except OSError:
                    pass

    # 按版本键清理 runs / exports / reports 子目录
    if resolved and run_key:
        _task, dirs = resolved
        for kind, root in dirs.items():
            versioned = root / run_key
            if versioned.is_dir():
                try:
                    shutil.rmtree(versioned)
                    removed_dirs.append(str(versioned))
                except OSError:
                    pass
            # 兼容旧版扁平文件：exports/reports 根下的同名文件
            if kind in {"exports", "reports"} and root.is_dir():
                for p in root.glob(f"{run_key}*"):
                    if p.is_file():
                        try:
                            p.unlink()
                            removed_files.append(str(p))
                        except OSError:
                            pass

    name = row.name
    model_id = row.id
    task_id = row.task_id
    db.delete(row)
    db.commit()

    if task_id:
        left = db.query(ModelRecord).filter(ModelRecord.task_id == task_id).count()
        if left == 0:
            task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
            if task and task.status in {"trained", "evaluated", "exported"}:
                pass

    return {
        "message": "已删除",
        "id": model_id,
        "name": name,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }
