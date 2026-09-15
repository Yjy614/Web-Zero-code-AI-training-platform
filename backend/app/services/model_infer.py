"""模型库推理：加载训练归档权重，对上传图片做检测/分割并返回可视化。"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image


class InferError(Exception):
    """推理业务错误（可转为 HTTP 400/422）。"""

    def __init__(self, message: str, code: str = "infer_failed") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _is_mock_weight(path: Path) -> bool:
    """演示占位权重无法真实推理。"""
    try:
        raw = path.read_bytes()[:64]
    except OSError:
        return False
    return raw.startswith(b"MOCK_") or b"MOCK_YOLO" in raw


def resolve_infer_weight(*, pt: Path | None, onnx: Path | None) -> tuple[Path, str]:
    """优先 PT，其次 ONNX。"""
    if pt and pt.is_file():
        if _is_mock_weight(pt):
            raise InferError(
                "当前权重文件无法用于推理，请使用训练产出的有效 PT 模型后重试。",
                code="mock_weight",
            )
        return pt, "pt"
    if onnx and onnx.is_file():
        return onnx, "onnx"
    raise InferError("找不到可用的 PT/ONNX 权重文件", code="no_weight")


def _encode_jpeg(rgb_or_bgr: Any, *, is_bgr: bool = True) -> tuple[str, int, int]:
    """将 numpy 图编码为 base64 JPEG；返回 (b64, w, h)。"""
    import numpy as np

    arr = np.asarray(rgb_or_bgr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise InferError("推理可视化图无效")
    if is_bgr:
        rgb = arr[:, :, ::-1]
    else:
        rgb = arr[:, :, :3]
    img = Image.fromarray(rgb.astype("uint8"), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, int(img.width), int(img.height)


def run_predict(
    weight_path: Path,
    image_bytes: bytes,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    task_type: str = "detect",
    model_format: str = "pt",
) -> dict[str, Any]:
    """
    对单张图片推理。
    返回 detections 列表 + image_base64 可视化（Ultralytics plot）。
    """
    if not image_bytes:
        raise InferError("图片内容为空")
    conf = float(max(0.01, min(0.99, conf)))
    iou = float(max(0.01, min(0.99, iou)))
    imgsz = int(max(320, min(1280, imgsz)))

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise InferError(
            "未安装 ultralytics，无法推理。请在后端环境安装依赖后重试。",
            code="no_ultralytics",
        ) from e

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise InferError("无法解析上传图片，请使用常见格式（jpg/png/webp）") from e

    try:
        model = YOLO(str(weight_path))
        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        raise InferError(f"推理失败：{e}") from e

    if not results:
        raise InferError("推理无结果")

    r0 = results[0]
    names = getattr(model, "names", None) or getattr(r0, "names", None) or {}
    if not isinstance(names, dict):
        try:
            names = dict(enumerate(names))
        except Exception:  # noqa: BLE001
            names = {}

    detections: list[dict[str, Any]] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        xyxy = r0.boxes.xyxy.cpu().tolist()
        confs = r0.boxes.conf.cpu().tolist()
        clss = r0.boxes.cls.cpu().tolist()
        polygons: list[list[list[float]] | None] = [None] * len(xyxy)
        if getattr(r0, "masks", None) is not None and getattr(r0.masks, "xy", None) is not None:
            for i, poly in enumerate(r0.masks.xy):
                if i >= len(polygons):
                    break
                try:
                    pts = [[float(p[0]), float(p[1])] for p in poly]
                    polygons[i] = pts if len(pts) >= 3 else None
                except Exception:  # noqa: BLE001
                    polygons[i] = None
        # 姿态关键点（像素坐标 xy + 可见性）
        kpts_list: list[list[list[float]] | None] = [None] * len(xyxy)
        if getattr(r0, "keypoints", None) is not None:
            try:
                kxy = r0.keypoints.xy.cpu().tolist()
                kconf = None
                if getattr(r0.keypoints, "conf", None) is not None:
                    kconf = r0.keypoints.conf.cpu().tolist()
                for i, pts in enumerate(kxy):
                    if i >= len(kpts_list):
                        break
                    row: list[list[float]] = []
                    for j, p in enumerate(pts):
                        x, y = float(p[0]), float(p[1])
                        v = 2.0
                        if kconf is not None and i < len(kconf) and j < len(kconf[i]):
                            c = float(kconf[i][j])
                            v = 2.0 if c >= 0.5 else (1.0 if c > 0.01 else 0.0)
                        row.append([round(x, 2), round(y, 2), v])
                    kpts_list[i] = row
            except Exception:  # noqa: BLE001
                pass

        for i, box in enumerate(xyxy):
            cid = int(clss[i])
            cname = str(names.get(cid, names.get(str(cid), f"class_{cid}")))
            item: dict[str, Any] = {
                "class_id": cid,
                "class_name": cname,
                "confidence": round(float(confs[i]), 4),
                "bbox_xyxy": [round(float(v), 2) for v in box],
            }
            if polygons[i]:
                item["polygon"] = polygons[i]
            if kpts_list[i]:
                item["keypoints"] = kpts_list[i]
            detections.append(item)

    # Ultralytics plot 输出 BGR ndarray
    plotted = r0.plot()
    image_b64, width, height = _encode_jpeg(plotted, is_bgr=True)

    return {
        "task_type": task_type or "detect",
        "model_format": model_format,
        "count": len(detections),
        "detections": detections,
        "image_base64": image_b64,
        "image_mime": "image/jpeg",
        "width": width,
        "height": height,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
    }
