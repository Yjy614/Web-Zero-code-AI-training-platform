"""
下载 Ultralytics coco8-pose 样例，并导入为本平台姿态数据集。

用法（在 backend 目录）：
  python scripts/import_coco8_pose.py
  python scripts/import_coco8_pose.py --owner demo
  python scripts/import_coco8_pose.py --with-weight
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.database import SessionLocal, init_db  # noqa: E402
from app.core.config import pretrained_root_path  # noqa: E402
from app.models.dataset import Dataset  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services import dataset_storage  # noqa: E402
from app.services.bootstrap import ensure_storage_dirs, seed_users  # noqa: E402
from app.services.pose_skeleton import coco17_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_NAME = "coco8_pose"


def _download_coco8_pose() -> Path:
    """通过 Ultralytics 拉取 coco8-pose，返回解压后的数据集根目录。"""
    from ultralytics.data.utils import check_det_dataset

    print("正在下载 / 校验 coco8-pose …")
    info = check_det_dataset("coco8-pose.yaml")
    # info['path'] 为数据集根
    root = Path(info["path"])
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 coco8-pose 目录：{root}")
    print(f"源数据目录：{root}")
    return root


def _collect_pairs(src_root: Path) -> list[tuple[Path, Path | None]]:
    """收集 images/** 与对应 labels/** 的成对文件。"""
    pairs: list[tuple[Path, Path | None]] = []
    images_root = src_root / "images"
    labels_root = src_root / "labels"
    if not images_root.is_dir():
        # 兼容扁平结构
        images_root = src_root
    for img in sorted(images_root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        # 相对 images 的子路径 → labels 同相对路径 .txt
        try:
            rel = img.relative_to(images_root)
        except ValueError:
            rel = Path(img.name)
        lbl = labels_root / rel.with_suffix(".txt")
        if not lbl.is_file():
            # 同 stem 扁平查找
            cand = labels_root / f"{img.stem}.txt"
            lbl_path = cand if cand.is_file() else None
        else:
            lbl_path = lbl
        pairs.append((img, lbl_path))
    return pairs


def _unique_name(dest_dir: Path, name: str) -> str:
    """避免同名覆盖：a.jpg / a_2.jpg …"""
    stem = Path(name).stem
    suf = Path(name).suffix.lower() or ".jpg"
    candidate = f"{stem}{suf}"
    n = 1
    while (dest_dir / candidate).exists():
        n += 1
        candidate = f"{stem}_{n}{suf}"
    return candidate


def import_into_platform(*, owner_username: str, also_weight: bool) -> int:
    ensure_storage_dirs()
    init_db()
    db = SessionLocal()
    try:
        seed_users(db)
        user = db.query(User).filter(User.username == owner_username).first()
        if not user:
            print(f"用户不存在：{owner_username}（请先启动一次后端以种子账号，或改用 --owner admin）")
            return 1

        # 同名则跳过入库，仍可刷新文件
        existing = (
            db.query(Dataset)
            .filter(
                Dataset.name == DATASET_NAME,
                Dataset.task_type == "pose",
                Dataset.owner_id == user.id,
            )
            .first()
        )

        src = _download_coco8_pose()
        pairs = _collect_pairs(src)
        if not pairs:
            print("源数据中没有图片，导入中止")
            return 1
        print(f"找到 {len(pairs)} 张图片")

        root = dataset_storage.ensure_dataset_dirs(
            user.username, DATASET_NAME, owner_id=user.id, task_type="pose"
        )
        img_dir = root / "images"
        lbl_dir = root / "labels"
        # 清空旧扁平标注，避免残留
        for d in (img_dir, lbl_dir):
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

        copied = 0
        for img, lbl in pairs:
            out_name = _unique_name(img_dir, img.name)
            shutil.copy2(img, img_dir / out_name)
            out_lbl = lbl_dir / f"{Path(out_name).stem}.txt"
            if lbl and lbl.is_file():
                shutil.copy2(lbl, out_lbl)
            else:
                out_lbl.write_text("", encoding="utf-8")
            copied += 1

        meta = dataset_storage.read_meta(root)
        meta["name"] = DATASET_NAME
        meta["task_type"] = "pose"
        meta["owner_id"] = user.id
        meta["owner_username"] = user.username
        meta["classes"] = ["person"]
        meta["pose"] = coco17_config()
        meta["source"] = "ultralytics/coco8-pose"
        dataset_storage.write_meta(root, meta)

        count = dataset_storage.count_images(root)
        if existing:
            existing.path = str(root)
            existing.classes_json = json.dumps(["person"], ensure_ascii=False)
            existing.image_count = count
            db.commit()
            print(f"已更新数据集 #{existing.id}「{DATASET_NAME}」→ {count} 张 · {root}")
        else:
            ds = Dataset(
                name=DATASET_NAME,
                path=str(root),
                task_type="pose",
                owner_id=user.id,
                classes_json=json.dumps(["person"], ensure_ascii=False),
                image_count=count,
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)
            print(f"已创建数据集 #{ds.id}「{DATASET_NAME}」→ {count} 张 · {root}")

        if also_weight:
            _ensure_pose_weight()
        else:
            print("提示：训练前请把 yolo11n-pose.pt 放到 pretrained/pose/，或加 --with-weight 自动下载")
        print("登录后：侧栏「姿态估计训练」或「数据集管理 → 姿态估计」即可看到 coco8_pose")
        return 0
    finally:
        db.close()


def _ensure_pose_weight() -> None:
    """下载常用轻量姿态权重到 pretrained/pose/。"""
    dest_dir = pretrained_root_path() / "pose"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = "yolo11n-pose.pt"
    dest = dest_dir / name
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"权重已存在：{dest}")
        return
    print(f"正在下载 {name} …")
    from ultralytics import YOLO

    # 触发 Ultralytics 缓存下载
    model = YOLO(name)
    src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
    if src is None or not Path(str(src)).is_file():
        # 回退：当前工作目录或默认权重名
        cand = Path(name)
        if not cand.is_file():
            from ultralytics.utils.downloads import attempt_download_asset

            cand = Path(attempt_download_asset(name))
        src = cand
    src = Path(src)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    print(f"权重已就绪：{dest} ({dest.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 coco8-pose 样例数据集")
    parser.add_argument("--owner", default="admin", help="归属用户名（默认 admin）")
    parser.add_argument("--with-weight", action="store_true", help="同时下载 yolo11n-pose.pt")
    args = parser.parse_args()
    return import_into_platform(owner_username=args.owner, also_weight=args.with_weight)


if __name__ == "__main__":
    raise SystemExit(main())
