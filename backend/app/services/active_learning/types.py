"""主动学习公共类型（任务无关）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Difficulty = Literal["easy", "hard", "empty"]


@dataclass
class DetectionBox:
    """单实例推理结果（像素坐标；分割含 polygon，姿态含 keypoints）。"""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    # 分割：像素多边形 [[x,y], ...]
    polygon: list[list[float]] | None = None
    # 姿态：像素关键点 [[x,y,v], ...]，v 为 YOLO 可见性 0/1/2
    keypoints: list[list[float]] | None = None


@dataclass
class ImageScore:
    """单张图的难易判定结果。"""

    image_name: str
    difficulty: Difficulty
    score: float  # 越大越「难」/越不确定
    max_conf: float | None = None
    top2_gap: float | None = None
    detection_count: int = 0
    reason: str = ""
    detections: list[DetectionBox] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ScreenSummary:
    """筛图汇总。"""

    total: int = 0
    easy_count: int = 0
    hard_count: int = 0
    empty_count: int = 0
    written_labels: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
