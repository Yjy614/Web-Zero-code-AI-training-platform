"""主动学习：基于框置信度的通用难易划分（detect / segment / pose 共用）。"""

from __future__ import annotations

from app.services.active_learning.types import DetectionBox, ImageScore

EASY_MIN_CONF = 0.55
HARD_MAX_CONF = 0.55
HARD_MIN_CONF = 0.20
TOP2_GAP_HARD = 0.15
WRITE_MIN_CONF = 0.25


def score_by_box_confidence(
    *,
    image_name: str,
    detections: list[DetectionBox],
    empty_reason: str = "未检出目标，建议人工确认是否漏检",
    uncertain_reason: str = "置信度不稳定或类别易混淆，建议优先复核",
    easy_reason: str = "模型较有把握，写入草稿后快速过一眼即可",
    low_reason: str = "检出置信度偏低，建议人工确认",
) -> ImageScore:
    """按最高框置信度与 top1/top2 间隔划分 easy / hard / empty。"""
    if not detections:
        return ImageScore(
            image_name=image_name,
            difficulty="empty",
            score=0.9,
            max_conf=None,
            top2_gap=None,
            detection_count=0,
            reason=empty_reason,
            detections=[],
        )

    confs = sorted((float(d.confidence) for d in detections), reverse=True)
    max_conf = confs[0]
    top2_gap = (confs[0] - confs[1]) if len(confs) >= 2 else 1.0

    uncertain = (HARD_MIN_CONF <= max_conf <= HARD_MAX_CONF) or (
        len(confs) >= 2 and top2_gap < TOP2_GAP_HARD and max_conf < 0.85
    )
    if uncertain:
        mid_pen = 1.0 - abs(max_conf - 0.5) * 2.0
        gap_pen = max(0.0, 1.0 - top2_gap / TOP2_GAP_HARD) if len(confs) >= 2 else 0.3
        score = round(0.55 * mid_pen + 0.45 * gap_pen, 4)
        return ImageScore(
            image_name=image_name,
            difficulty="hard",
            score=max(0.1, min(1.0, score)),
            max_conf=round(max_conf, 4),
            top2_gap=round(top2_gap, 4),
            detection_count=len(detections),
            reason=uncertain_reason,
            detections=detections,
        )

    if max_conf >= EASY_MIN_CONF:
        return ImageScore(
            image_name=image_name,
            difficulty="easy",
            score=round(1.0 - max_conf, 4),
            max_conf=round(max_conf, 4),
            top2_gap=round(top2_gap, 4),
            detection_count=len(detections),
            reason=easy_reason,
            detections=detections,
        )

    return ImageScore(
        image_name=image_name,
        difficulty="hard",
        score=0.8,
        max_conf=round(max_conf, 4),
        top2_gap=round(top2_gap, 4),
        detection_count=len(detections),
        reason=low_reason,
        detections=detections,
    )
