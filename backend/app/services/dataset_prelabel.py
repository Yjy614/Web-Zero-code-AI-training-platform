"""AI 预标注：用已标注图短训临时模型，再推理写入未标注图；权重用完即删。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.dataset import BBox, Point2D, PolygonInstance
from app.services import dataset_annotate, dataset_storage
from app.services.task_storage import list_weight_files, normalize_weight_task_type
from app.services.weight_resolve import resolve_pretrained_weight

# 一期门槛与超参（固定，不做复杂 UI）
MIN_LABELED = 20
MIN_UNLABELED = 1
PRELABEL_EPOCHS = 50
PRELABEL_IMGSZ = 640
PRELABEL_BATCH = 8
PRELABEL_CONF = 0.35
META_KEY = "prelabel_last"


def _image_is_labeled(root: Path, name: str, task_type: str) -> bool:
    """按任务类型判断是否已有标注。"""
    tt = (task_type or "detect").strip().lower()
    if tt == "segment":
        return bool(dataset_annotate.read_polygons(root, name))
    if tt == "pose":
        return bool(dataset_annotate.read_poses(root, name))
    return bool(dataset_annotate.read_annotations(root, name))

def analyze_prelabel(root: Path, task_type: str = "detect") -> dict[str, Any]:
    """统计可预标注的已标注 / 未标注活跃图。"""
    tt = normalize_weight_task_type(task_type) if task_type else "detect"
    dataset_storage.migrate_physical_removed_to_excluded(root)
    labeled: list[str] = []
    unlabeled: list[str] = []
    for name in dataset_storage.list_images(root, include_removed=False):
        if _image_is_labeled(root, name, tt):
            labeled.append(name)
        else:
            unlabeled.append(name)
    meta = dataset_storage.read_meta(root)
    classes = meta.get("classes") if isinstance(meta.get("classes"), list) else []
    last = meta.get(META_KEY) if isinstance(meta.get(META_KEY), dict) else {}
    return {
        "labeled": labeled,
        "unlabeled": unlabeled,
        "labeled_count": len(labeled),
        "unlabeled_count": len(unlabeled),
        "class_count": len(classes),
        "classes": classes,
        "can_prelabel": len(classes) >= 1
        and len(labeled) >= MIN_LABELED
        and len(unlabeled) >= MIN_UNLABELED,
        "min_labeled": MIN_LABELED,
        "min_unlabeled": MIN_UNLABELED,
        "last_written": list(last.get("written") or []),
        "task_type": tt,
    }


def prelabel_block_reason(info: dict[str, Any]) -> str | None:
    """不可启动时的中文原因；可启动返回 None。"""
    if int(info.get("class_count") or 0) < 1:
        return "请先添加至少一个类别"
    if int(info.get("labeled_count") or 0) < MIN_LABELED:
        return f"至少需要 {MIN_LABELED} 张已标注图片（当前 {info.get('labeled_count', 0)}）"
    if int(info.get("unlabeled_count") or 0) < MIN_UNLABELED:
        return "没有未标注图片可预标注"
    return None


def resolve_prelabel_weight(task_type: str = "detect") -> Path:
    """选取预标注起点权重：优先 nano，否则仓库内任意 .pt。"""
    tt = normalize_weight_task_type(task_type)
    if tt == "pose":
        preferred = ("yolo11n-pose.pt", "yolov8n-pose.pt", "yolo11s-pose.pt", "yolov8s-pose.pt")
        for name in preferred:
            try:
                return resolve_pretrained_weight(name, tt)
            except FileNotFoundError:
                continue
        files = list_weight_files(tt)
        for item in files:
            name = str(item.get("name") or "")
            if name.lower().endswith(".pt"):
                return resolve_pretrained_weight(name, tt)
        raise FileNotFoundError("未找到姿态预训练权重，请上传 *-pose.pt 到 pretrained/pose/")
    if tt == "segment":
        preferred = (
            "yolov8n-seg.pt",
            "yolo11n-seg.pt",
            "yolov8s-seg.pt",
            "yolo11s-seg.pt",
            "yolov8m-seg.pt",
        )
        miss_msg = "权重仓库中没有可用的分割 .pt，请管理员先上传基础权重（如 yolov8n-seg.pt）"
    else:
        preferred = ("yolov8n.pt", "yolo11n.pt", "yolov8s.pt", "yolo11s.pt")
        miss_msg = "权重仓库中没有可用的检测 .pt，请管理员先上传基础权重（如 yolov8n.pt）"
    for name in preferred:
        try:
            return resolve_pretrained_weight(name, tt)
        except FileNotFoundError:
            continue
    files = list_weight_files(tt)
    for item in files:
        name = str(item.get("name") or "")
        if name.lower().endswith(".pt"):
            try:
                return resolve_pretrained_weight(name, tt)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(miss_msg)


def _tmp_dir(root: Path) -> Path:
    return root / ".prelabel_tmp"


def cleanup_prelabel_tmp(root: Path) -> None:
    """删除预标注临时目录（公开接口，供 runner finally 调用）。"""
    tmp = _tmp_dir(root)
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


def _cleanup_tmp(root: Path) -> None:
    cleanup_prelabel_tmp(root)


def _build_temp_yolo_dataset(
    root: Path,
    labeled: list[str],
    classes: list[str],
) -> Path:
    """在临时目录构建仅含已标注图的 YOLO 数据集，返回 data.yaml。"""
    tmp = _tmp_dir(root)
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    train_img = tmp / "images" / "train"
    val_img = tmp / "images" / "val"
    train_lbl = tmp / "labels" / "train"
    val_lbl = tmp / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    names = list(labeled)
    # 至少留 1 张验证；样本少时验证集取 1 张
    if len(names) >= 5:
        val_n = max(1, len(names) // 10)
    else:
        val_n = 1 if len(names) > 1 else 0
    val_set = set(names[:val_n]) if val_n else set()
    train_set = [n for n in names if n not in val_set]
    if not train_set:
        train_set = list(names)
        val_set = set()

    def _copy_one(name: str, img_dir: Path, lbl_dir: Path) -> None:
        src_img = root / "images" / name
        src_lbl = dataset_storage.label_path_for(root, name)
        if not src_img.is_file() or not src_lbl.is_file():
            return
        shutil.copy2(src_img, img_dir / name)
        shutil.copy2(src_lbl, lbl_dir / f"{Path(name).stem}.txt")

    for name in train_set:
        _copy_one(name, train_img, train_lbl)
    for name in val_set:
        _copy_one(name, val_img, val_lbl)
    # 无验证集时用训练集顶上，避免 ultralytics 报错
    if not any(val_img.iterdir()):
        for name in train_set[:1]:
            _copy_one(name, val_img, val_lbl)

    yaml_path = tmp / "data.yaml"
    # path 用绝对路径，避免 cwd 影响
    lines = [
        f"path: {tmp.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    for i, c in enumerate(classes):
        lines.append(f"  {i}: {c}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return (
        max(0.0, min(1.0, xc / w)),
        max(0.0, min(1.0, yc / h)),
        max(1e-6, min(1.0, bw / w)),
        max(1e-6, min(1.0, bh / h)),
    )


def _predict_boxes(model: Any, image_path: Path, class_count: int) -> tuple[list[BBox], int]:
    """推理一张图，返回 (写入框, 因低置信跳过数)。"""
    results = model.predict(
        source=str(image_path),
        conf=PRELABEL_CONF,
        verbose=False,
        imgsz=PRELABEL_IMGSZ,
    )
    if not results:
        return [], 0
    r0 = results[0]
    skipped = 0
    boxes: list[BBox] = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return [], 0
    h, w = int(r0.orig_shape[0]), int(r0.orig_shape[1])
    for b in r0.boxes:
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        if conf < PRELABEL_CONF:
            skipped += 1
            continue
        cls_id = int(b.cls[0]) if b.cls is not None else 0
        if cls_id < 0 or cls_id >= class_count:
            skipped += 1
            continue
        xyxy = b.xyxy[0].tolist()
        xc, yc, bw, bh = _xyxy_to_yolo(xyxy[0], xyxy[1], xyxy[2], xyxy[3], w, h)
        boxes.append(BBox(class_id=cls_id, x_center=xc, y_center=yc, width=bw, height=bh))
    return boxes, skipped


def _predict_polygons(
    model: Any, image_path: Path, class_count: int
) -> tuple[list[PolygonInstance], int]:
    """分割推理一张图，返回 (多边形列表, 低置信跳过数)。"""
    results = model.predict(
        source=str(image_path),
        conf=PRELABEL_CONF,
        verbose=False,
        imgsz=PRELABEL_IMGSZ,
        retina_masks=True,
    )
    if not results:
        return [], 0
    r0 = results[0]
    skipped = 0
    polys: list[PolygonInstance] = []
    if r0.boxes is None or len(r0.boxes) == 0:
        return [], 0
    h, w = int(r0.orig_shape[0]), int(r0.orig_shape[1])
    mask_xy = getattr(getattr(r0, "masks", None), "xy", None)
    for i, b in enumerate(r0.boxes):
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        if conf < PRELABEL_CONF:
            skipped += 1
            continue
        cls_id = int(b.cls[0]) if b.cls is not None else 0
        if cls_id < 0 or cls_id >= class_count:
            skipped += 1
            continue
        pts_raw = None
        if mask_xy is not None and i < len(mask_xy):
            pts_raw = mask_xy[i]
        if pts_raw is None or len(pts_raw) < 3:
            skipped += 1
            continue
        points: list[Point2D] = []
        for p in pts_raw:
            points.append(
                Point2D(
                    x=max(0.0, min(1.0, float(p[0]) / max(w, 1))),
                    y=max(0.0, min(1.0, float(p[1]) / max(h, 1))),
                )
            )
        # 过密抽稀，避免标签过大
        if len(points) > 80:
            step = max(1, len(points) // 80)
            points = points[::step]
            if len(points) < 3:
                skipped += 1
                continue
        polys.append(PolygonInstance(class_id=cls_id, points=points))
    return polys, skipped


def save_prelabel_meta(root: Path, written: list[str], job_id: int) -> None:
    meta = dataset_storage.read_meta(root)
    meta[META_KEY] = {
        "written": written,
        "job_id": job_id,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    dataset_storage.write_meta(root, meta)


def commit_prelabel_as_user(root: Path) -> dict[str, Any]:
    """
    将本轮 AI 预标注正式确认为用户标注：保留 labels，仅清除可撤销记录。
    训练成功后调用，之后「清除本轮 AI 预标注」不再可用。
    """
    meta = dataset_storage.read_meta(root)
    last = meta.get(META_KEY) if isinstance(meta.get(META_KEY), dict) else None
    if not last:
        return {"committed": 0, "message": "无待确认的预标注"}
    written = [str(x) for x in (last.get("written") or []) if x]
    meta.pop(META_KEY, None)
    dataset_storage.write_meta(root, meta)
    n = len(written)
    return {"committed": n, "message": f"已确认 {n} 张预标注为用户标注"}


def revert_prelabel(root: Path) -> dict[str, Any]:
    """清除本轮 AI 预标注写入的标注文件。"""
    meta = dataset_storage.read_meta(root)
    last = meta.get(META_KEY) if isinstance(meta.get(META_KEY), dict) else {}
    written = [str(x) for x in (last.get("written") or []) if x]
    removed = 0
    for name in written:
        path = dataset_storage.label_path_for(root, name)
        if path.exists():
            path.unlink(missing_ok=True)
            removed += 1
    meta.pop(META_KEY, None)
    dataset_storage.write_meta(root, meta)
    return {"reverted": removed, "message": f"已清除本轮 AI 预标注 {removed} 张"}


def run_prelabel_job(
    *,
    root: Path,
    job_id: int,
    task_type: str,
    device: str,
    update_progress,
    is_cancelled,
) -> dict[str, Any]:
    """
    执行预标注全流程。update_progress(progress, message) 写 Job 状态。
    成功/失败/取消均清理临时目录与权重。
    """
    info = analyze_prelabel(root, task_type)
    reason = prelabel_block_reason(info)
    if reason:
        raise ValueError(reason)

    labeled: list[str] = info["labeled"]
    unlabeled: list[str] = info["unlabeled"]
    classes: list[str] = info["classes"]
    weight = resolve_prelabel_weight(task_type)
    is_segment = (task_type or "detect") == "segment"

    update_progress(5, "准备预标注数据…")
    if is_cancelled():
        raise InterruptedError("预标注已取消")

    yaml_path = _build_temp_yolo_dataset(root, labeled, classes)
    runs_dir = _tmp_dir(root) / "runs"

    written: list[str] = []
    skipped_labeled = 0  # 未标注列表本身已排除已标注
    low_conf_skipped = 0
    YOLO = None
    try:
        from ultralytics import YOLO as _YOLO

        YOLO = _YOLO
    except ImportError as e:
        raise RuntimeError("未安装 ultralytics，无法预标注") from e

    model = None
    try:
        update_progress(0, f"快速训练中（{PRELABEL_EPOCHS} epochs）…")
        model = YOLO(str(weight))

        def on_fit_epoch_end(trainer) -> None:  # noqa: ANN001
            if is_cancelled():
                trainer.stop = True
                return
            raw = int(getattr(trainer, "epoch", 0)) + 1
            ep = min(raw, PRELABEL_EPOCHS)
            # 训练阶段：进度 = 已训练轮数 / 总轮数
            pct = round(ep / PRELABEL_EPOCHS * 100, 1)
            update_progress(min(100.0, pct), f"快速训练中 epoch {ep}/{PRELABEL_EPOCHS}")

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=str(yaml_path),
            epochs=PRELABEL_EPOCHS,
            batch=PRELABEL_BATCH,
            imgsz=PRELABEL_IMGSZ,
            device=device,
            project=str(runs_dir.parent),
            name=runs_dir.name,
            exist_ok=True,
            plots=False,
            save=True,
            verbose=False,
            workers=0,
        )

        if is_cancelled():
            raise InterruptedError("预标注已取消")

        best = runs_dir / "weights" / "best.pt"
        last = runs_dir / "weights" / "last.pt"
        weights_path = best if best.is_file() else last
        if not weights_path.is_file():
            raise FileNotFoundError("预标注短训未产出权重，请检查已标注数据质量")

        # 训练已按 epoch 走到 100%；推理阶段保持 100%，仅更新文案与完成态
        update_progress(100, "正在对未标注图片推理…")
        pred_model = YOLO(str(weights_path))
        total = len(unlabeled)
        for i, name in enumerate(unlabeled):
            if is_cancelled():
                raise InterruptedError("预标注已取消")
            # 再次确认仍无人工标注（避免过程中被写入）
            if _image_is_labeled(root, name, task_type):
                skipped_labeled += 1
                continue
            img_path = root / "images" / name
            if not img_path.is_file():
                continue
            if is_segment:
                polys, skipped = _predict_polygons(pred_model, img_path, len(classes))
                low_conf_skipped += skipped
                if polys:
                    dataset_annotate.write_polygons(root, name, polys)
                    written.append(name)
            else:
                boxes, skipped = _predict_boxes(pred_model, img_path, len(classes))
                low_conf_skipped += skipped
                if boxes:
                    dataset_annotate.write_annotations(root, name, boxes)
                    written.append(name)
            update_progress(100, f"推理写入 {i + 1}/{total}")

        save_prelabel_meta(root, written, job_id)
        unit = "个实例" if is_segment else "个框"
        result = {
            "written": len(written),
            "written_files": written,
            "skipped_existing": skipped_labeled,
            "low_conf_skipped": low_conf_skipped,
            "labeled_used": len(labeled),
            "unlabeled_total": len(unlabeled),
            "message": (
                f"预标注完成：写入 {len(written)} 张；"
                f"跳过已有标注 {skipped_labeled} 张；"
                f"低置信未写入 {low_conf_skipped} {unit}"
            ),
        }
        update_progress(100, result["message"])
        return result
    finally:
        # 权重与临时数据一律删干净
        model = None
        _cleanup_tmp(root)


def run_prelabel_mock(
    *,
    root: Path,
    job_id: int,
    update_progress,
    is_cancelled,
    task_type: str = "detect",
) -> dict[str, Any]:
    """演示模式：不真实训练，给未标注图写占位标注。"""
    import time

    info = analyze_prelabel(root, task_type)
    reason = prelabel_block_reason(info)
    if reason:
        raise ValueError(reason)
    unlabeled: list[str] = info["unlabeled"]
    classes: list[str] = info["classes"]
    class_id = 0 if classes else 0
    is_segment = (task_type or "detect") == "segment"
    written: list[str] = []
    update_progress(0, f"快速训练中（{PRELABEL_EPOCHS} epochs）…")
    demo_steps = 10
    for step in range(1, demo_steps + 1):
        if is_cancelled():
            raise InterruptedError("预标注已取消")
        time.sleep(0.2)
        ep = max(1, int(round(step / demo_steps * PRELABEL_EPOCHS)))
        pct = round(ep / PRELABEL_EPOCHS * 100, 1)
        update_progress(min(100.0, pct), f"快速训练中 epoch {ep}/{PRELABEL_EPOCHS}")
    update_progress(100, "正在对未标注图片推理…")
    for i, name in enumerate(unlabeled):
        if is_cancelled():
            raise InterruptedError("预标注已取消")
        if _image_is_labeled(root, name, task_type):
            continue
        if is_segment:
            dataset_annotate.write_polygons(
                root,
                name,
                [
                    PolygonInstance(
                        class_id=class_id,
                        points=[
                            Point2D(x=0.50, y=0.32),
                            Point2D(x=0.68, y=0.50),
                            Point2D(x=0.50, y=0.68),
                            Point2D(x=0.32, y=0.50),
                        ],
                    )
                ],
            )
        else:
            dataset_annotate.write_annotations(
                root,
                name,
                [BBox(class_id=class_id, x_center=0.5, y_center=0.5, width=0.35, height=0.35)],
            )
        written.append(name)
        update_progress(100, f"写入 {i + 1}/{len(unlabeled)}")
    save_prelabel_meta(root, written, job_id)
    _cleanup_tmp(root)
    msg = f"预标注完成：写入 {len(written)} 张"
    update_progress(100, msg)
    return {
        "written": len(written),
        "written_files": written,
        "skipped_existing": 0,
        "low_conf_skipped": 0,
        "labeled_used": info["labeled_count"],
        "unlabeled_total": len(unlabeled),
        "message": msg,
        "demo": True,
    }
