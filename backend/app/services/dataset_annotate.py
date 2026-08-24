"""YOLO 标注读写：检测 bbox / 实例分割 polygon。"""

from __future__ import annotations

from pathlib import Path

from app.schemas.dataset import BBox, Point2D, PolygonInstance
from app.services.dataset_storage import label_path_for


def read_annotations(root: Path, image_name: str) -> list[BBox]:
    """读取某张图的 YOLO 检测 txt 标注（5 列）。"""
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
    """写入 YOLO 检测 txt；空列表则删除标注文件。"""
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


def read_polygons(root: Path, image_name: str) -> list[PolygonInstance]:
    """读取 YOLO-seg txt：class_id x1 y1 x2 y2 ...（归一化）。"""
    path = label_path_for(root, image_name)
    if not path.exists():
        return []
    polys: list[PolygonInstance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 至少 class + 3 个点（6 个坐标）
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        points = [
            Point2D(x=min(1.0, max(0.0, coords[i])), y=min(1.0, max(0.0, coords[i + 1])))
            for i in range(0, len(coords), 2)
        ]
        if len(points) < 3:
            continue
        polys.append(PolygonInstance(class_id=class_id, points=points))
    return polys


def write_polygons(root: Path, image_name: str, polygons: list[PolygonInstance]) -> None:
    """写入 YOLO-seg txt；空列表则删除标注文件。"""
    path = label_path_for(root, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not polygons:
        if path.exists():
            path.unlink()
        return
    lines: list[str] = []
    for poly in polygons:
        if len(poly.points) < 3:
            continue
        coords = " ".join(f"{p.x:.6f} {p.y:.6f}" for p in poly.points)
        lines.append(f"{poly.class_id} {coords}")
    if not lines:
        if path.exists():
            path.unlink()
        return
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iter_label_stems(root: Path) -> list[str]:
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        return []
    return sorted(p.stem for p in labels_dir.glob("*.txt") if p.is_file())


def _remap_class_id(cid: int, removed_sorted_desc: list[int]) -> int | None:
    """删除若干类别后，将旧 class_id 映射为新 id；被删类别返回 None。"""
    if cid in removed_sorted_desc:
        return None
    new_id = cid
    for rid in removed_sorted_desc:
        if cid > rid:
            new_id -= 1
    return new_id


def purge_and_remap_class_ids(
    root: Path,
    *,
    task_type: str = "detect",
    removed_indices: list[int] | None = None,
    class_count: int | None = None,
) -> dict:
    """
    删除指定类别的标注，并将更大的 class_id 前移。
    若提供 class_count，额外清除 class_id >= class_count 的孤儿标注。
    """
    removed = sorted({int(i) for i in (removed_indices or []) if int(i) >= 0}, reverse=True)
    is_segment = (task_type or "detect") == "segment"
    touched = 0
    removed_ann = 0

    for stem in _iter_label_stems(root):
        # label 文件名与图片 stem 一致；用任意扩展名拼回去给读写函数
        image_name = f"{stem}.jpg"
        if is_segment:
            items = read_polygons(root, image_name)
            if not items and not label_path_for(root, image_name).exists():
                continue
            next_items: list[PolygonInstance] = []
            changed = False
            for poly in items:
                mapped = _remap_class_id(poly.class_id, removed)
                if mapped is None:
                    removed_ann += 1
                    changed = True
                    continue
                if class_count is not None and mapped >= class_count:
                    removed_ann += 1
                    changed = True
                    continue
                if mapped != poly.class_id:
                    changed = True
                next_items.append(PolygonInstance(class_id=mapped, points=poly.points))
            if changed:
                write_polygons(root, image_name, next_items)
                touched += 1
        else:
            items_b = read_annotations(root, image_name)
            if not items_b and not label_path_for(root, image_name).exists():
                continue
            next_boxes: list[BBox] = []
            changed = False
            for box in items_b:
                mapped = _remap_class_id(box.class_id, removed)
                if mapped is None:
                    removed_ann += 1
                    changed = True
                    continue
                if class_count is not None and mapped >= class_count:
                    removed_ann += 1
                    changed = True
                    continue
                if mapped != box.class_id:
                    changed = True
                next_boxes.append(
                    BBox(
                        class_id=mapped,
                        x_center=box.x_center,
                        y_center=box.y_center,
                        width=box.width,
                        height=box.height,
                    )
                )
            if changed:
                write_annotations(root, image_name, next_boxes)
                touched += 1

    return {"files_touched": touched, "annotations_removed": removed_ann}
