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
) -> str:
    """
    生成不重复模型名：首次用 base，其后 base_2、base_3…
    会把历史名（base-best、base-时间戳）和归档目录里已有文件一并视为占用。
    """
    import re

    base = (base or "model").strip() or "model"
    existing: set[str] = set()
    for (n,) in db.query(ModelRecord.name).filter(ModelRecord.owner_id == owner_id).all():
        if n:
            existing.add(str(n))
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for p in archive_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".onnx"}:
                existing.add(p.stem)

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
    """优先模型权重旁的 .onnx，其次任务导出目录。"""
    candidates: list[Path] = []
    pt = find_pt_path(row)
    if pt:
        candidates.append(pt.with_suffix(".onnx"))
    resolved = artifact_dirs_for_model(db, row)
    if resolved:
        task, dirs = resolved
        export_dir = dirs["exports"]
        if export_dir.is_dir():
            # 与该模型同名戳的导出优先
            stem = pt.stem if pt else ""
            if stem:
                candidates.append(export_dir / f"{stem}.onnx")
            candidates.append(export_dir / f"{task.name}.onnx")
            candidates.append(export_dir / f"{task.name}_demo.onnx")
            candidates.extend(sorted(export_dir.glob("*.onnx")))
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def delete_model_artifacts(db: Session, row: ModelRecord) -> dict:
    """
    删除单条模型记录及其归档权重 / 旁路 ONNX。
    同任务其他模型保留；不再清空整个 runs（避免误删其它版本）。
    """
    removed_files: list[str] = []
    pt = find_pt_path(row)
    if pt and pt.is_file():
        try:
            pt.unlink()
            removed_files.append(str(pt))
        except OSError:
            pass
        side = pt.with_suffix(".onnx")
        if side.is_file():
            try:
                side.unlink()
                removed_files.append(str(side))
            except OSError:
                pass

    name = row.name
    model_id = row.id
    task_id = row.task_id
    db.delete(row)
    db.commit()

    # 若任务已无其它模型记录，可把状态回落（不强制清 model_path，评估仍可用最近 runs）
    if task_id:
        left = db.query(ModelRecord).filter(ModelRecord.task_id == task_id).count()
        if left == 0:
            task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
            if task and task.status in {"trained", "evaluated", "exported"}:
                # 仍保留 model_path 便于继续评估；仅作轻量标记
                pass

    return {
        "message": "已删除",
        "id": model_id,
        "name": name,
        "removed_files": removed_files,
    }
