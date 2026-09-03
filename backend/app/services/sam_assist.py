"""SAM2 标注辅助：点选提示 → 掩膜 → 归一化多边形。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import pretrained_root_path

logger = logging.getLogger(__name__)

# 固定权重文件名（与 Ultralytics 资源一致）
SAM2_WEIGHT_NAME = "sam2_b.pt"


class SamAssistError(Exception):
    """SAM 辅助标注业务错误。"""

    def __init__(self, message: str, code: str = "sam_failed") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def sam_weights_dir() -> Path:
    """SAM 权重目录：pretrained/sam/（与 detect/segment 训练权重隔离）。"""
    return pretrained_root_path() / "sam"


def ensure_sam_dir() -> Path:
    """确保 SAM 目录存在。"""
    d = sam_weights_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_sam2_weight() -> Path:
    """解析本地 sam2_b.pt；缺失时给出中文提示（不自动联网下载）。"""
    ensure_sam_dir()
    path = sam_weights_dir() / SAM2_WEIGHT_NAME
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    raise SamAssistError(
        f"未找到 SAM2 权重：{path}。请将 {SAM2_WEIGHT_NAME} 放到 pretrained/sam/ 目录。",
        code="sam_weight_missing",
    )


@lru_cache(maxsize=1)
def _load_sam_model(weight_str: str):
    """缓存加载 SAM2，避免每次点选都重新读盘。"""
    from ultralytics import SAM

    return SAM(weight_str)


def clear_sam_cache() -> None:
    """清理模型缓存（测试或热更新权重时用）。"""
    _load_sam_model.cache_clear()


def _mask_to_normalized_polygon(
    mask: np.ndarray, width: int, height: int, *, max_points: int = 80
) -> list[dict[str, float]]:
    """二值掩膜 → 归一化多边形顶点（尽量简化点数）。"""
    try:
        import cv2
    except ImportError as e:
        raise SamAssistError(
            "缺少 opencv-python，无法从掩膜提取轮廓。请安装 opencv-python-headless。",
            code="no_opencv",
        ) from e

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.squeeze()
    binary = (m > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SamAssistError("SAM 未生成有效轮廓", code="empty_mask")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        raise SamAssistError("SAM 掩膜过小，请换位置再点", code="tiny_mask")

    peri = cv2.arcLength(contour, True)
    # 按周长自适应近似；再按点数上限抽稀
    eps = max(1.0, 0.004 * peri)
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]
    if len(pts) < 3:
        raise SamAssistError("轮廓点数不足，请换位置再点", code="few_points")

    w = max(int(width), 1)
    h = max(int(height), 1)
    out: list[dict[str, float]] = []
    for x, y in pts:
        out.append(
            {
                "x": round(float(np.clip(x / w, 0.0, 1.0)), 6),
                "y": round(float(np.clip(y / h, 0.0, 1.0)), 6),
            }
        )
    return out


def predict_polygon_from_point(
    image_path: Path,
    *,
    x_norm: float,
    y_norm: float,
    positive: bool = True,
) -> dict[str, Any]:
    """
    在归一化坐标 (x_norm, y_norm) 上做 SAM2 正点/负点提示，返回归一化多边形。
    """
    if not image_path.is_file():
        raise SamAssistError("图片不存在", code="no_image")

    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
    except Exception as e:  # noqa: BLE001
        raise SamAssistError(f"无法读取图片：{e}", code="bad_image") from e

    px = float(x_norm * (width - 1))
    py = float(y_norm * (height - 1))
    label = 1 if positive else 0

    weight = resolve_sam2_weight()
    try:
        model = _load_sam_model(str(weight.resolve()))
        results = model.predict(
            source=rgb,
            points=[[px, py]],
            labels=[label],
            verbose=False,
        )
    except SamAssistError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("SAM2 推理失败")
        raise SamAssistError(f"SAM2 推理失败：{e}") from e

    if not results:
        raise SamAssistError("SAM 无结果", code="no_result")
    r0 = results[0]
    if r0.masks is None or len(r0.masks) == 0:
        raise SamAssistError("未分割出目标，请换位置再点", code="no_mask")

    # 取第一张掩膜（点提示通常一条）
    data = r0.masks.data
    if hasattr(data, "cpu"):
        mask = data[0].cpu().numpy()
    else:
        mask = np.asarray(data[0])
    # masks 可能是模型输入尺寸，优先用 xy 多边形
    if getattr(r0.masks, "xy", None) is not None and len(r0.masks.xy) > 0:
        poly_xy = r0.masks.xy[0]
        if poly_xy is not None and len(poly_xy) >= 3:
            pts = [
                {
                    "x": round(float(np.clip(p[0] / width, 0.0, 1.0)), 6),
                    "y": round(float(np.clip(p[1] / height, 0.0, 1.0)), 6),
                }
                for p in poly_xy
            ]
            # 过密则抽稀
            if len(pts) > 80:
                idx = np.linspace(0, len(pts) - 1, 80, dtype=int)
                pts = [pts[i] for i in idx]
            return {
                "points": pts,
                "width": width,
                "height": height,
                "model": SAM2_WEIGHT_NAME,
            }

    points = _mask_to_normalized_polygon(mask, width, height)
    return {
        "points": points,
        "width": width,
        "height": height,
        "model": SAM2_WEIGHT_NAME,
    }
