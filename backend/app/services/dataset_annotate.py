"""YOLO 检测标注读写。"""

from __future__ import annotations

from pathlib import Path

from app.schemas.dataset import BBox
from app.services.dataset_storage import label_path_for


def read_annotations(root: Path, image_name: str) -> list[BBox]:
    """读取某张图的 YOLO txt 标注。"""
    path = label_path_for(root, image_name)
    if not path.exists():
        return []
    boxes: list[BBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append(
            BBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return boxes


def write_annotations(root: Path, image_name: str, boxes: list[BBox]) -> None:
    """写入 YOLO txt；空列表则删除标注文件。"""
    path = label_path_for(root, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        if path.exists():
            path.unlink()
        return
    lines = [
        f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}" for b in boxes
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
