"""划分数据集并生成 YOLO data.yaml（train / val / test）。"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from app.services.dataset_storage import list_images, read_meta, write_meta


def prepare_yolo_split(
    dataset_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float | None = None,
    seed: int = 42,
) -> dict:
    """
    将 images/labels 按比例划分到 train/val/test，并写 data.yaml。
    test_ratio 为空时按 1 - train - val 计算。
    """
    images = list_images(dataset_root)
    if len(images) < 1:
        raise ValueError("数据集没有可用图片，无法划分")

    tr = max(0.05, float(train_ratio))
    vr = max(0.0, float(val_ratio))
    if test_ratio is None:
        te = max(0.0, 1.0 - tr - vr)
    else:
        te = max(0.0, float(test_ratio))
    total = tr + vr + te
    if total <= 0:
        raise ValueError("划分比例无效")
    tr, vr, te = tr / total, vr / total, te / total

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n = len(shuffled)

    # 尽量保证各集合至少有样本（图片足够时）
    n_train = max(1, int(round(n * tr)))
    n_val = int(round(n * vr)) if vr > 0 else 0
    n_test = n - n_train - n_val
    if te > 0 and n >= 3 and n_test < 1:
        # 从 train 匀一点给 test
        if n_train > 1:
            n_train -= 1
            n_test = 1
    if vr > 0 and n >= 2 and n_val < 1:
        if n_train > 1:
            n_train -= 1
            n_val = 1
    # 修正溢出
    while n_train + n_val + n_test > n and n_train > 1:
        n_train -= 1
    n_test = max(0, n - n_train - n_val)

    train_list = shuffled[:n_train]
    val_list = shuffled[n_train : n_train + n_val]
    test_list = shuffled[n_train + n_val :]

    splits = ("train", "val", "test")
    for split in splits:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 清理旧划分
    for split in splits:
        for sub in ("images", "labels"):
            d = dataset_root / sub / split
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    def _copy_pair(name: str, split: str) -> None:
        src_img = dataset_root / "images" / name
        dst_img = dataset_root / "images" / split / name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        stem = Path(name).stem
        src_lbl = dataset_root / "labels" / f"{stem}.txt"
        dst_lbl = dataset_root / "labels" / split / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")

    for name in train_list:
        _copy_pair(name, "train")
    for name in val_list:
        _copy_pair(name, "val")
    for name in test_list:
        _copy_pair(name, "test")

    meta = read_meta(dataset_root)
    classes = meta.get("classes") or []
    if isinstance(classes, str):
        try:
            classes = json.loads(classes)
        except json.JSONDecodeError:
            classes = []

    yaml_text = _build_data_yaml(dataset_root, classes, has_test=len(test_list) > 0)
    (dataset_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    meta["split"] = {
        "train": len(train_list),
        "val": len(val_list),
        "test": len(test_list),
        "train_ratio": round(tr, 4),
        "val_ratio": round(vr, 4),
        "test_ratio": round(te, 4),
    }
    write_meta(dataset_root, meta)

    return {
        "train": len(train_list),
        "val": len(val_list),
        "test": len(test_list),
        "classes": classes,
        "yaml": str(dataset_root / "data.yaml"),
    }


def _build_data_yaml(dataset_root: Path, classes: list, *, has_test: bool) -> str:
    names = classes if classes else ["object"]
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    return "\n".join(lines) + "\n"
