"""Segment 主动学习策略：框置信度 + 掩膜完整性。"""

from __future__ import annotations

from typing import Any

from app.schemas.dataset import Point2D, PolygonInstance
from app.services.active_learning.conf_score import WRITE_MIN_CONF, score_by_box_confidence
from app.services.active_learning.types import DetectionBox, ImageScore


class SegmentActiveLearnStrategy:
    """实例分割难易划分 + YOLO-seg 多边形写出。"""

    task_type = "segment"

    def score_image(
        self,
        *,
        image_name: str,
        detections: list[DetectionBox],
        image_size: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageScore:
        _ = image_size, extra
        base = score_by_box_confidence(
            image_name=image_name,
            detections=detections,
            empty_reason="未检出实例，建议人工确认是否漏检",
            uncertain_reason="分割置信度不稳定，建议优先复核轮廓",
            easy_reason="模型较有把握，已写多边形草稿，快速过一眼即可",
            low_reason="分割置信度偏低，建议人工确认",
        )
        if base.difficulty == "empty":
            return base

        # 有框但缺掩膜 / 顶点过少 → 抬为难例
        weak_mask = 0
        for d in detections:
            poly = d.polygon or []
            if len(poly) < 3:
                weak_mask += 1
        if weak_mask > 0 and base.difficulty == "easy":
            return ImageScore(
                image_name=image_name,
                difficulty="hard",
                score=max(base.score, 0.65),
                max_conf=base.max_conf,
                top2_gap=base.top2_gap,
                detection_count=len(detections),
                reason=f"有 {weak_mask} 个实例缺少有效轮廓，建议人工补全",
                detections=detections,
            )
        if weak_mask > 0 and base.difficulty == "hard" and not base.reason:
            base.reason = "掩膜不完整或置信不稳，建议优先复核"
        return base

    def detections_to_label_payload(
        self,
        score: ImageScore,
        *,
        image_width: int,
        image_height: int,
    ) -> list[PolygonInstance]:
        """像素多边形 → YOLO-seg 归一化顶点。"""
        w = max(1, int(image_width))
        h = max(1, int(image_height))
        out: list[PolygonInstance] = []
        for d in score.detections:
            if float(d.confidence) < WRITE_MIN_CONF:
                continue
            raw = d.polygon or []
            if len(raw) < 3:
                continue
            points: list[Point2D] = []
            for p in raw:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                points.append(
                    Point2D(
                        x=min(1.0, max(0.0, float(p[0]) / w)),
                        y=min(1.0, max(0.0, float(p[1]) / h)),
                    )
                )
            if len(points) < 3:
                continue
            out.append(PolygonInstance(class_id=int(d.class_id), points=points))
        return out
