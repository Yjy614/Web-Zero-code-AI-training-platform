"""任务类型常量（检测 / 分割）。"""

from __future__ import annotations

TASK_TYPES = ("detect", "segment")

TASK_TYPE_LABELS = {
    "detect": "目标检测",
    "segment": "实例分割",
}


def normalize_task_type(task_type: str | None) -> str:
    """校验并规范化任务类型。"""
    t = (task_type or "detect").strip().lower()
    if t not in TASK_TYPES:
        raise ValueError(f"不支持的任务类型：{task_type}，可选：{', '.join(TASK_TYPES)}")
    return t
