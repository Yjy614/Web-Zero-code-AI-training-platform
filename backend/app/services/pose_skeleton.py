"""姿态关键点骨架模板：COCO-17 与自定义 N。"""

from __future__ import annotations

from typing import Any

# COCO 人体 17 点（Ultralytics YOLO-Pose 默认）
COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# 0-based 骨架连线（与常见 COCO skeleton 一致）
COCO17_SKELETON = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]

# 左右翻转时关键点下标映射
COCO17_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def coco17_config() -> dict[str, Any]:
    """返回 COCO-17 完整姿态配置。"""
    n = len(COCO17_NAMES)
    return {
        "template": "coco17",
        "kpt_shape": [n, 3],
        "kpt_names": list(COCO17_NAMES),
        "skeleton": [list(e) for e in COCO17_SKELETON],
        "flip_idx": list(COCO17_FLIP_IDX),
    }


def custom_config(
    *,
    kpt_names: list[str] | None = None,
    kpt_count: int | None = None,
    skeleton: list[list[int]] | None = None,
    flip_idx: list[int] | None = None,
) -> dict[str, Any]:
    """构建自定义关键点配置。"""
    names = [str(x).strip() for x in (kpt_names or []) if str(x).strip()]
    if not names:
        n = int(kpt_count or 0)
        if n < 1:
            raise ValueError("自定义姿态至少需要 1 个关键点")
        if n > 64:
            raise ValueError("自定义关键点数过多（最多 64）")
        names = [f"kpt_{i}" for i in range(n)]
    else:
        if len(names) > 64:
            raise ValueError("自定义关键点数过多（最多 64）")
    n = len(names)
    edges: list[list[int]] = []
    for e in skeleton or []:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a, b = int(e[0]), int(e[1])
        if 0 <= a < n and 0 <= b < n and a != b:
            edges.append([a, b])
    flips = list(flip_idx) if flip_idx is not None else list(range(n))
    if len(flips) != n:
        flips = list(range(n))
    else:
        flips = [int(x) if 0 <= int(x) < n else i for i, x in enumerate(flips)]
    return {
        "template": "custom",
        "kpt_shape": [n, 3],
        "kpt_names": names,
        "skeleton": edges,
        "flip_idx": flips,
    }


def normalize_pose_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """规范化 meta.pose；非法或缺省时回退 COCO-17。"""
    if not isinstance(raw, dict):
        return coco17_config()
    template = str(raw.get("template") or "").strip().lower()
    if template == "coco17" or (not template and not raw.get("kpt_names") and not raw.get("kpt_shape")):
        # 显式 coco17，或空配置
        if template == "coco17":
            return coco17_config()
    try:
        names = raw.get("kpt_names")
        shape = raw.get("kpt_shape")
        kpt_count = None
        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
            kpt_count = int(shape[0])
        if template == "custom" or names or kpt_count:
            cfg = custom_config(
                kpt_names=list(names) if isinstance(names, list) else None,
                kpt_count=kpt_count,
                skeleton=raw.get("skeleton") if isinstance(raw.get("skeleton"), list) else None,
                flip_idx=raw.get("flip_idx") if isinstance(raw.get("flip_idx"), list) else None,
            )
            if template == "coco17":
                cfg["template"] = "coco17"
            return cfg
    except (TypeError, ValueError):
        pass
    return coco17_config()


def pose_config_from_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """从数据集 meta 读取姿态配置。"""
    if not isinstance(meta, dict):
        return coco17_config()
    return normalize_pose_config(meta.get("pose") if isinstance(meta.get("pose"), dict) else None)


def kpt_count(cfg: dict[str, Any]) -> int:
    shape = cfg.get("kpt_shape") or [17, 3]
    try:
        return max(1, int(shape[0]))
    except (TypeError, ValueError, IndexError):
        return 17
