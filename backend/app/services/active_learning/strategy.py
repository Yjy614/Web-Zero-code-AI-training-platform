"""主动学习策略抽象：各任务类型实现各自的难易划分。"""

from __future__ import annotations

from typing import Any, Protocol

from app.services.active_learning.types import DetectionBox, ImageScore

SUPPORTED_TASK_TYPES = frozenset({"detect", "segment", "pose"})


class ActiveLearnStrategy(Protocol):
    """任务相关策略接口（Detect / Segment / Pose 各自实现）。"""

    task_type: str

    def score_image(
        self,
        *,
        image_name: str,
        detections: list[DetectionBox],
        image_size: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageScore:
        """根据推理结果判定 easy / hard / empty。"""
        ...

    def detections_to_label_payload(
        self,
        score: ImageScore,
        *,
        image_width: int,
        image_height: int,
    ) -> Any:
        """将 easy 样本的检测转为可写入标注的数据结构（任务相关）。"""
        ...


def get_strategy(task_type: str) -> ActiveLearnStrategy:
    """按任务类型获取策略。"""
    tt = (task_type or "detect").strip().lower()
    if tt == "detect":
        from app.services.active_learning.detect_strategy import DetectActiveLearnStrategy

        return DetectActiveLearnStrategy()
    if tt == "segment":
        from app.services.active_learning.segment_strategy import SegmentActiveLearnStrategy

        return SegmentActiveLearnStrategy()
    if tt == "pose":
        from app.services.active_learning.pose_strategy import PoseActiveLearnStrategy

        return PoseActiveLearnStrategy()
    raise NotImplementedError(f"不支持的任务类型：{tt}")
