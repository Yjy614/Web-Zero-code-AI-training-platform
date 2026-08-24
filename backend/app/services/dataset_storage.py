"""数据集存储路径与元数据工具。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from app.core.config import storage_root_path
from app.core.task_types import normalize_task_type

SAFE_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff\-]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset_name(name: str) -> str:
    """校验数据集名称，避免路径穿越。"""
    name = name.strip()
    if not name or not SAFE_NAME_RE.match(name):
        raise ValueError("数据集名称仅允许中文、字母、数字、下划线与短横线")
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("数据集名称非法")
    return name


def owner_storage_key(username: str) -> str:
    """用户级目录名：使用登录用户名（直观、可读）。"""
    name = (username or "").strip()
    if not name or not SAFE_NAME_RE.match(name):
        raise ValueError("用户名仅允许中文、字母、数字、下划线与短横线，且不能为空")
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("用户名非法")
    return name


def owner_dir(username: str, task_type: str = "detect") -> Path:
    """返回某用户下指定任务类型的数据集父目录。"""
    tt = normalize_task_type(task_type)
    return storage_root_path() / "datasets" / tt / owner_storage_key(username)


def dataset_dir(username: str, name: str, task_type: str = "detect") -> Path:
    """返回用户隔离后的数据集根目录。"""
    return owner_dir(username, task_type) / name


def ensure_dataset_dirs(
    username: str, name: str, *, owner_id: int = 0, task_type: str = "detect"
) -> Path:
    """创建 images / labels / removed 目录，并写入空 meta。"""
    tt = normalize_task_type(task_type)
    root = dataset_dir(username, name, tt)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "removed" / "images").mkdir(parents=True, exist_ok=True)
    (root / "removed" / "labels").mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        write_meta(
            root,
            {
                "name": name,
                "task_type": tt,
                "owner_id": owner_id,
                "owner_username": owner_storage_key(username),
                "classes": [],
                "excluded": [],
            },
        )
    return root


def write_meta(root: Path, data: dict) -> None:
    """写入 meta.json。"""
    (root / "meta.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(root: Path) -> dict:
    """读取 meta.json。"""
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_excluded(root: Path) -> set[str]:
    """读取不参与训练的排除文件名集合。"""
    meta = read_meta(root)
    raw = meta.get("excluded") or []
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw if x}


def set_excluded(root: Path, names: set[str]) -> None:
    """写入排除列表。"""
    meta = read_meta(root)
    meta["excluded"] = sorted(names)
    write_meta(root, meta)


def list_all_image_files(root: Path) -> list[str]:
    """列出 images/ 下全部图片文件名（含已排除）。"""
    images_dir = root / "images"
    if not images_dir.exists():
        return []
    return sorted(
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_images(root: Path, *, include_removed: bool = False) -> list[str]:
    """
    列出图片文件名。
    - 默认：仅返回参与训练的活跃图（未在 excluded 中）
    - include_removed=True：返回 images/ 下全部（含排除项）
    """
    names = list_all_image_files(root)
    if include_removed:
        return names
    excluded = get_excluded(root)
    return [n for n in names if n not in excluded]


def count_images(root: Path) -> int:
    """数据集内图片总数（含不参与训练的排除项；已永久删除的不计）。"""
    return len(list_all_image_files(root))


def count_active_images(root: Path) -> int:
    """参与训练的活跃图片数量（不含排除项）。"""
    return len(list_images(root, include_removed=False))


def label_path_for(root: Path, image_name: str, *, removed: bool = False) -> Path:
    """图片对应的 YOLO txt 标注路径；removed=True 时指向 removed/labels（兼容旧数据）。"""
    stem = Path(image_name).stem
    if removed:
        return root / "removed" / "labels" / f"{stem}.txt"
    return root / "labels" / f"{stem}.txt"


def migrate_physical_removed_to_excluded(root: Path) -> int:
    """
    兼容旧逻辑：把 removed/images 中的文件迁回 images/，并写入 excluded。
    返回迁回张数。
    """
    removed_img = root / "removed" / "images"
    if not removed_img.exists():
        return 0
    excluded = get_excluded(root)
    moved = 0
    for path in list(removed_img.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        name = path.name
        target = root / "images" / name
        if target.exists():
            # 活跃区已有同名：删除 removed 副本，仍标记排除
            path.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved += 1
        excluded.add(name)
        src_label = label_path_for(root, name, removed=True)
        if src_label.exists():
            dst_label = label_path_for(root, name, removed=False)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            if not dst_label.exists():
                shutil.move(str(src_label), str(dst_label))
            else:
                src_label.unlink(missing_ok=True)
    set_excluded(root, excluded)
    return moved


def is_safe_image_name(name: str) -> bool:
    """校验图片文件名。"""
    if not name or "/" in name or "\\" in name or ".." in name:
        return False
    return Path(name).suffix.lower() in IMAGE_EXTS


def is_user_scoped_path(
    path: Path, username: str, name: str, task_type: str = "detect"
) -> bool:
    """判断 path 是否已是 datasets/<task_type>/<username>/<name> 结构。"""
    try:
        return path.resolve() == dataset_dir(username, name, task_type).resolve()
    except OSError:
        return False


def migrate_dataset_to_user_scope(
    old_path: str | Path,
    username: str,
    name: str,
    *,
    owner_id: int = 0,
    task_type: str = "detect",
) -> Path:
    """
    将旧版扁平目录或 user_<id> 目录迁移到 datasets/<task_type>/<username>/<name>。
    若目标已存在则保留目标；返回最终路径。
    """
    tt = normalize_task_type(task_type)
    target = dataset_dir(username, name, tt)
    target.parent.mkdir(parents=True, exist_ok=True)
    src = Path(old_path)

    if target.exists():
        if src.exists() and src.resolve() != target.resolve():
            _merge_dir(src, target)
            shutil.rmtree(src, ignore_errors=True)
        return target

    if src.exists():
        shutil.move(str(src), str(target))
    else:
        ensure_dataset_dirs(username, name, owner_id=owner_id, task_type=tt)

    meta = read_meta(target)
    meta["name"] = name
    meta["task_type"] = tt
    meta["owner_id"] = owner_id
    meta["owner_username"] = owner_storage_key(username)
    if "classes" not in meta:
        meta["classes"] = []
    write_meta(target, meta)
    return target


def _merge_dir(src: Path, dst: Path) -> None:
    """将 src 内容合并进 dst（同名文件跳过）。"""
    if not src.exists():
        return
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        dest_file = dst / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if not dest_file.exists():
            shutil.move(str(item), str(dest_file))
