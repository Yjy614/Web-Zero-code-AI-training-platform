"""主动学习临时数据集：识别、删除、悬空清理。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.train import Job, TrainTask
from app.models.user import User
from app.services import dataset_storage
from app.services.active_learning import session_store

# 仍占用临时集的会话状态（未并入 / 未取消）
ACTIVE_SESSION_STATUSES = frozenset({"created", "uploaded", "screening", "screened"})
# 临时集应已删除的终态
TERMINAL_SESSION_STATUSES = frozenset({"merged", "retraining", "done", "abandoned", "error"})


def is_active_learn_staging(ds: Dataset) -> bool:
    """是否为主动学习临时集（未并入前不应出现在数据集管理列表）。"""
    name = (ds.name or "").strip()
    if name.startswith("AL_"):
        return True
    if not ds.path:
        return False
    try:
        meta = dataset_storage.read_meta(Path(ds.path))
    except Exception:  # noqa: BLE001
        return False
    return bool(meta.get("active_learn"))


def delete_staging_dataset(db: Session, ds: Dataset) -> None:
    """删除临时集：关联 Job / TrainTask → DB 记录 → 磁盘目录。"""
    tasks = db.query(TrainTask).filter(TrainTask.dataset_id == ds.id).all()
    for t in tasks:
        db.query(Job).filter(Job.task_id == t.id).delete(synchronize_session=False)
        db.delete(t)
    root = Path(ds.path) if ds.path else None
    db.delete(ds)
    db.commit()
    if root is not None and root.exists():
        shutil.rmtree(root, ignore_errors=True)


def abandon_session(db: Session, user: User, session_id: str) -> dict[str, Any]:
    """中断主动学习：删除临时集；进行中会话标记为 abandoned。"""
    session = session_store.load_session(user.id, session_id)
    status = str(session.get("status") or "")

    staging_id = session.get("staging_dataset_id")
    if staging_id:
        ds = db.query(Dataset).filter(Dataset.id == int(staging_id)).first()
        if ds:
            if user.role != "admin" and ds.owner_id != user.id:
                raise PermissionError("无权删除该临时数据集")
            delete_staging_dataset(db, ds)

    # 若筛图 Job 仍在跑，尽量标为取消
    job_id = session.get("job_id")
    if job_id:
        job = db.query(Job).filter(Job.id == int(job_id)).first()
        if job and job.status in {"pending", "running"}:
            job.status = "cancelled"
            job.message = "主动学习已取消"
            db.commit()

    session["staging_dataset_id"] = None
    session["staging_dataset_name"] = None
    # 已并入/续训中：只清临时集，不改终态
    if status in {"merged", "retraining", "done", "abandoned"}:
        session_store.save_session(user.id, session)
        return {"session": session, "message": "临时数据集已清理"}

    session["status"] = "abandoned"
    session_store.save_session(user.id, session)
    return {"session": session, "message": "已取消本轮主动学习，临时数据集已删除"}


def cleanup_orphaned_staging(db: Session, user: User) -> dict[str, Any]:
    """
    清理悬空临时集：
    - 无任何「进行中」会话引用的 AL 临时集
    - 已终态会话仍残留的临时集
    """
    q = db.query(Dataset)
    if user.role != "admin":
        q = q.filter(Dataset.owner_id == user.id)
    rows = q.all()

    # 进行中会话引用的临时集 ID
    active_staging_ids: set[int] = set()
    for s in session_store.list_sessions(user.id, limit=100):
        st = str(s.get("status") or "")
        sid = s.get("staging_dataset_id")
        if sid is None:
            continue
        try:
            iid = int(sid)
        except (TypeError, ValueError):
            continue
        if st in ACTIVE_SESSION_STATUSES:
            active_staging_ids.add(iid)
        elif st in TERMINAL_SESSION_STATUSES:
            # 终态仍挂着临时集 → 一并清掉
            ds = db.query(Dataset).filter(Dataset.id == iid).first()
            if ds and is_active_learn_staging(ds):
                if user.role == "admin" or ds.owner_id == user.id:
                    delete_staging_dataset(db, ds)
                    s["staging_dataset_id"] = None
                    s["staging_dataset_name"] = None
                    session_store.save_session(user.id, s)

    removed: list[int] = []
    for ds in rows:
        if not is_active_learn_staging(ds):
            continue
        if ds.id in active_staging_ids:
            continue
        # 重新查，可能已被上面删除
        alive = db.query(Dataset).filter(Dataset.id == ds.id).first()
        if not alive:
            continue
        delete_staging_dataset(db, alive)
        removed.append(int(ds.id))

    return {"removed": removed, "count": len(removed)}


def abandon_other_active_sessions(
    db: Session,
    user: User,
    *,
    keep_session_id: str | None = None,
    model_id: int | None = None,
) -> int:
    """新建会话前：取消同一用户其它进行中的主动学习（默认同模型）。"""
    n = 0
    for s in session_store.list_sessions(user.id, limit=50):
        if keep_session_id and str(s.get("id")) == str(keep_session_id):
            continue
        if model_id is not None and int(s.get("model_id") or 0) != int(model_id):
            continue
        if str(s.get("status") or "") not in ACTIVE_SESSION_STATUSES:
            continue
        try:
            abandon_session(db, user, str(s["id"]))
            n += 1
        except Exception:  # noqa: BLE001
            continue
    return n
