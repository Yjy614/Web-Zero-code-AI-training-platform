"""训练策略预设：面向零代码用户的三档方案（学习率等藏在策略内）。"""

from __future__ import annotations

from typing import Any, Literal

TrainStrategyId = Literal["stable", "fast", "finetune"]

DEFAULT_TRAIN_STRATEGY: TrainStrategyId = "stable"

# 策略定义：Ultralytics YOLO 常用超参子集
TRAIN_STRATEGIES: dict[str, dict[str, Any]] = {
    "stable": {
        "id": "stable",
        "label": "稳健（默认）",
        "description": "沿用 YOLO 常用学习率与线性衰减，适合大多数工业检测首训。",
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": False,
        "warmup_epochs": 3.0,
    },
    "fast": {
        "id": "fast",
        "label": "快速收敛",
        "description": "余弦学习率 + 较短预热，前期下降更快；曲线抖动大时可改回稳健。",
        "lr0": 0.01,
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 1.0,
    },
    "finetune": {
        "id": "finetune",
        "label": "细调（小学习率）",
        "description": "更小初始学习率，适合已有不错权重、继续微调或小样本防过拟合。",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 3.0,
    },
}


def normalize_train_strategy(value: str | None) -> str:
    """非法或空值回退为默认策略。"""
    key = (value or "").strip()
    if key in TRAIN_STRATEGIES:
        return key
    return DEFAULT_TRAIN_STRATEGY


def resolve_train_hparams(strategy: str | None) -> dict[str, Any]:
    """解析策略为 model.train(...) 可用的超参。"""
    key = normalize_train_strategy(strategy)
    preset = TRAIN_STRATEGIES[key]
    return {
        "train_strategy": key,
        "lr0": float(preset["lr0"]),
        "lrf": float(preset["lrf"]),
        "cos_lr": bool(preset["cos_lr"]),
        "warmup_epochs": float(preset["warmup_epochs"]),
    }


def strategy_choices() -> list[dict[str, Any]]:
    """前端展示用的策略列表（不含内部数值时可按需裁剪）。"""
    return [
        {
            "id": p["id"],
            "label": p["label"],
            "description": p["description"],
        }
        for p in TRAIN_STRATEGIES.values()
    ]
