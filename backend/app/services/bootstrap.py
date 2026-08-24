"""种子数据与存储目录初始化。"""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings, pretrained_root_path, storage_root_path
from app.core.security import hash_password
from app.models.dataset import Dataset
from app.models.train import TrainTask
from app.models.user import User
from app.services import dataset_storage
from app.services.task_storage import WEIGHT_TASK_TYPES


def ensure_storage_dirs() -> None:
    """创建用户数据目录与系统预训练权重目录。"""
    root = storage_root_path()
    for rel in (
        "datasets/detect",
        "datasets/segment",
        "runs/detect",
        "runs/segment",
        "exports/detect",
        "exports/segment",
        "reports/detect",
        "reports/segment",
        "models/detect",
        "models/segment",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)

    pre = pretrained_root_path()
    for tt in WEIGHT_TASK_TYPES:
        (pre / tt).mkdir(parents=True, exist_ok=True)

    # 纠正：误放在 pretrained/<tt>/<用户>/ 下的训练产物迁回 storage/models
    _recover_misplaced_trained_models()


def seed_users(db: Session) -> None:
    """写入配置中的种子账号（已存在则跳过）。"""
    settings = get_settings()
    for item in settings.seed_users:
        username = item.get("username")
        password = item.get("password")
        role = item.get("role", "user")
        if not username or not password:
            continue
        exists = db.query(User).filter(User.username == username).first()
        if exists:
            continue
        db.add(User(username=username, password_hash=hash_password(password), role=role))
    db.commit()


def username_by_id(db: Session, owner_id: int) -> str:
    """按用户 id 解析目录用用户名；找不到时回退 user_<id>。"""
    user = db.query(User).filter(User.id == owner_id).first()
    if user and user.username:
        return dataset_storage.owner_storage_key(user.username)
    return f"user_{int(owner_id)}"


def migrate_datasets_to_user_scope(db: Session) -> None:
    """
    将旧版扁平路径或 user_<id> 路径
    迁移为 datasets/<task_type>/<username>/<name>，并回写 DB.path。
    """
    # 先把磁盘上 user_<id> 目录整体改名为用户名
    _migrate_owner_dirs_on_disk(db)

    rows = db.query(Dataset).all()
    changed = False
    for ds in rows:
        uname = username_by_id(db, ds.owner_id)
        tt = ds.task_type or "detect"
        current = Path(ds.path)
        if dataset_storage.is_user_scoped_path(current, uname, ds.name, tt) and current.exists():
            continue
        new_path = dataset_storage.migrate_dataset_to_user_scope(
            ds.path,
            uname,
            ds.name,
            owner_id=ds.owner_id,
            task_type=tt,
        )
        if str(new_path) != ds.path:
            ds.path = str(new_path)
            changed = True
    # 训练产物路径若写在 model_path 里含 user_<id>，一并替换
    for task in db.query(TrainTask).all():
        uname = username_by_id(db, task.owner_id)
        old_key = f"user_{task.owner_id}"
        if task.model_path and old_key in task.model_path.replace("\\", "/"):
            task.model_path = task.model_path.replace(old_key, uname).replace(
                f"user_{task.owner_id}",
                uname,
            )
            changed = True
    if changed:
        db.commit()


def _migrate_owner_dirs_on_disk(db: Session) -> None:
    """将 datasets/runs/exports/reports 下的 user_<id> 目录重命名为用户名。"""
    root = storage_root_path()
    kinds = ("datasets", "runs", "exports", "reports")
    task_types = ("detect", "segment")
    for user in db.query(User).all():
        old_key = f"user_{user.id}"
        try:
            new_key = dataset_storage.owner_storage_key(user.username)
        except ValueError:
            continue
        if old_key == new_key:
            continue
        for kind in kinds:
            for tt in task_types:
                old_dir = root / kind / tt / old_key
                new_dir = root / kind / tt / new_key
                if not old_dir.exists():
                    continue
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                if not new_dir.exists():
                    shutil.move(str(old_dir), str(new_dir))
                else:
                    # 目标已存在：合并后删旧目录
                    _merge_tree(old_dir, new_dir)
                    shutil.rmtree(old_dir, ignore_errors=True)


def _merge_tree(src: Path, dst: Path) -> None:
    """合并目录树，同名文件跳过。"""
    if not src.exists():
        return
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.move(str(item), str(dest))


def _recover_misplaced_trained_models() -> None:
    """
    将误放在 pretrained/<task_type>/<用户名>/ 下的训练归档迁回 storage/models。
    pretrained/<task_type>/ 下的直接 .pt 文件（如 yolo11n.pt）保留不动。
    """
    pre = pretrained_root_path()
    models_root = storage_root_path() / "models"
    for tt in WEIGHT_TASK_TYPES:
        pre_tt = pre / tt
        if not pre_tt.is_dir():
            continue
        dest_tt = models_root / tt
        dest_tt.mkdir(parents=True, exist_ok=True)
        for item in list(pre_tt.iterdir()):
            if not item.is_dir():
                continue
            dest = dest_tt / item.name
            dest.mkdir(parents=True, exist_ok=True)
            _merge_tree(item, dest)
            # 迁完后删除空用户子目录
            shutil.rmtree(item, ignore_errors=True)


def repair_model_record_paths(db: Session) -> None:
    """若模型库记录仍指向 pretrained/<tt>/<用户>/…，改写为 storage/models 路径。"""
    from app.models.train import ModelRecord

    pre = pretrained_root_path().resolve()
    models_root = (storage_root_path() / "models").resolve()
    changed = False
    for row in db.query(ModelRecord).all():
        raw = (row.path or "").strip()
        if not raw:
            continue
        rel: Path | None = None
        p = Path(raw)
        try:
            base = p.resolve() if p.exists() else p
            rel = base.relative_to(pre)
        except ValueError:
            parts = list(Path(raw.replace("\\", "/")).parts)
            idx = next((i for i, x in enumerate(parts) if x.lower() == "pretrained"), -1)
            if idx >= 0 and idx + 1 < len(parts):
                rel = Path(*parts[idx + 1 :])
        if rel is None or len(rel.parts) < 3:
            # detect/demo/ties_train.pt 才是用户归档；detect/yolo11n.pt 不改
            continue
        new_path = models_root / rel
        if str(Path(row.path)) != str(new_path):
            row.path = str(new_path)
            changed = True
    if changed:
        db.commit()
