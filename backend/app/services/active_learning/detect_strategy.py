"""Detect 主动学习策略：置信度 + 不确定性（top1/top2 间隔）。"""

from __future__ import annotations

from typing import Any

from app.schemas.dataset import BBox
from app.services.active_learning.conf_score import WRITE_MIN_CONF, score_by_box_confidence
from app.services.active_learning.types import DetectionBox, ImageScore


class DetectActiveLearnStrategy:
    """目标检测难易划分 + YOLO 归一化框写出。"""

    task_type = "detect"

    def score_image(
        self,
        *,
        image_name: str,
        detections: list[DetectionBox],
        image_size: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageScore:
        _ = image_size, extra
        return score_by_box_confidence(image_name=image_name, detections=detections)

    def detections_to_label_payload(
        self,
        score: ImageScore,
        *,
        image_width: int,
        image_height: int,
    ) -> list[BBox]:
        """像素 xyxy → YOLO 归一化框；过滤过低置信。"""
        w = max(1, int(image_width))
        h = max(1, int(image_height))
        boxes: list[BBox] = []
        for d in score.detections:
            if float(d.confidence) < WRITE_MIN_CONF:
                continue
            x1, y1, x2, y2 = [float(v) for v in d.bbox_xyxy]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            nw = bw / w
            nh = bh / h
            boxes.append(
                BBox(
                    class_id=int(d.class_id),
                    x_center=min(1.0, max(0.0, xc)),
                    y_center=min(1.0, max(0.0, yc)),
                    width=min(1.0, max(1e-6, nw)),
                    height=min(1.0, max(1e-6, nh)),
                )
            )
        return boxes
