"""训练任务 / Job Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TrainConfig(BaseModel):
    """向导配置步超参。"""

    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.4)
    test_ratio: float = Field(default=0.1, ge=0.0, le=0.4)
    epochs: int = Field(default=50, ge=1, le=500)
    batch: int = Field(default=8, ge=1, le=128)
    imgsz: int = Field(default=640, ge=320, le=1280)
    # 训练策略：stable / fast / finetune（学习率等由策略映射，不直接暴露）
    train_strategy: str = Field(default="stable")
    device: str = Field(default="cpu")
    pretrained_weight: str = Field(default="")  # 权重仓库中的文件名
    augment: bool = True

    @model_validator(mode="after")
    def _normalize_split(self) -> "TrainConfig":
        """保证 train+val+test=1；测试集取剩余（可为 0）；策略非法时回退默认。"""
        from app.services.train_strategy import normalize_train_strategy

        tr = round(float(self.train_ratio), 2)
        vr = round(float(self.val_ratio), 2)
        if tr + vr > 1.0:
            vr = max(0.05, round(1.0 - tr, 2))
        te = max(0.0, round(1.0 - tr - vr, 2))
        self.train_ratio = tr
        self.val_ratio = vr
        self.test_ratio = te
        self.train_strategy = normalize_train_strategy(self.train_strategy)
        return self


class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    dataset_id: int


class TaskUpdate(BaseModel):
    step: int | None = None
    status: str | None = None
    config: TrainConfig | None = None


class TaskOut(BaseModel):
    id: int
    name: str
    owner_id: int
    dataset_id: int
    status: str
    step: int
    config: dict[str, Any] = Field(default_factory=dict)
    model_path: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class JobOut(BaseModel):
    id: int
    task_id: int
    type: str
    status: str
    progress: float
    message: str
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ExportRequest(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["pt", "onnx"])


class AiAdviceOut(BaseModel):
    """评估步 AI 优化建议。"""

    advice: str
    source: str = Field(description="llm | fallback")
    context: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WeightItem(BaseModel):
    name: str
    path: str = ""
    size: int
    task_type: str = "detect"


class ModelOut(BaseModel):
    id: int
    name: str
    path: str
    task_type: str
    owner_id: int
    task_id: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    # 是否有可下载的 .pt / .onnx
    has_pt: bool = False
    has_onnx: bool = False
    # 训练血缘：原数据集与续训父权重
    dataset_id: int | None = None
    dataset_name: str | None = None
    parent_model_id: int | None = None
    parent_weight: str | None = None
    parent_weight_label: str | None = None


class DetectItemOut(BaseModel):
    """单条检测/分割结果。"""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    polygon: list[list[float]] | None = None


class PredictOut(BaseModel):
    """模型推理试用结果。"""

    task_type: str
    model_format: str
    count: int
    detections: list[DetectItemOut] = Field(default_factory=list)
    image_base64: str
    image_mime: str = "image/jpeg"
    width: int
    height: int
    conf: float
    iou: float
    imgsz: int
