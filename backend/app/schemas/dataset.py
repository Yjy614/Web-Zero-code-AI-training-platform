"""数据集相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="数据集名称")


class DatasetOut(BaseModel):
    id: int
    name: str
    path: str
    task_type: str
    owner_id: int
    classes: list[str] = []
    image_count: int  # 图片总数（含排除项）
    active_count: int = 0  # 参与训练的活跃图片数
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ClassesUpdate(BaseModel):
    classes: list[str] = Field(default_factory=list)


class BBox(BaseModel):
    """YOLO 归一化框：中心点 + 宽高，均为 0~1。"""

    class_id: int = Field(..., ge=0)
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)


class AnnotationPayload(BaseModel):
    boxes: list[BBox] = Field(default_factory=list)


class AnnotationOut(BaseModel):
    image: str
    boxes: list[BBox]


class CleanResult(BaseModel):
    kept: int
    removed: int
    removed_files: list[str] = Field(default_factory=list)
    message: str = ""


class ImageItem(BaseModel):
    name: str
    has_label: bool = False
    status: str = "active"  # active | removed（removed=不参与训练，文件仍保留）


class ImageNamesPayload(BaseModel):
    """选中图片文件名列表。"""

    names: list[str] = Field(default_factory=list)
