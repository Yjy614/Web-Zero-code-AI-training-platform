"""数据集相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="数据集名称")
    task_type: str = Field(default="detect", description="任务类型：detect | segment | pose")
    # 姿态数据集可选：创建时写入 meta.pose（缺省 COCO-17）
    pose_template: str | None = Field(default=None, description="coco17 | custom")
    pose_kpt_names: list[str] | None = Field(default=None, description="自定义关键点名称")
    pose_kpt_count: int | None = Field(default=None, ge=1, le=64, description="自定义关键点数")
    pose_skeleton: list[list[int]] | None = Field(default=None, description="骨架连线 0-based")


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
    # 本次删除的旧类别下标：删除后清掉对应标注，并将更大的 class_id 前移
    removed_indices: list[int] = Field(default_factory=list)


class BBox(BaseModel):
    """YOLO 归一化框：中心点 + 宽高，均为 0~1。"""

    class_id: int = Field(..., ge=0)
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)


class Point2D(BaseModel):
    """归一化坐标点（0~1）。"""

    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)


class PolygonInstance(BaseModel):
    """YOLO-seg 多边形实例：至少 3 个归一化顶点。"""

    class_id: int = Field(..., ge=0)
    points: list[Point2D] = Field(..., min_length=3)


class Keypoint(BaseModel):
    """YOLO-Pose 归一化关键点：x/y 与可见性 v（0 未标 / 1 遮挡 / 2 可见）。"""

    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    v: int = Field(default=0, ge=0, le=2)


class PoseInstance(BaseModel):
    """YOLO-Pose 实例：bbox + 固定长度关键点序列。"""

    class_id: int = Field(..., ge=0)
    x_center: float = Field(..., ge=0, le=1)
    y_center: float = Field(..., ge=0, le=1)
    width: float = Field(..., gt=0, le=1)
    height: float = Field(..., gt=0, le=1)
    keypoints: list[Keypoint] = Field(default_factory=list)


class PoseSkeletonOut(BaseModel):
    """数据集姿态骨架配置。"""

    template: str
    kpt_shape: list[int]
    kpt_names: list[str]
    skeleton: list[list[int]] = Field(default_factory=list)
    flip_idx: list[int] = Field(default_factory=list)


class PoseSkeletonUpdate(BaseModel):
    template: str = Field(default="coco17", description="coco17 | custom")
    kpt_names: list[str] | None = None
    kpt_count: int | None = Field(default=None, ge=1, le=64)
    skeleton: list[list[int]] | None = None
    flip_idx: list[int] | None = None


class AnnotationPayload(BaseModel):
    boxes: list[BBox] = Field(default_factory=list)
    polygons: list[PolygonInstance] = Field(default_factory=list)
    poses: list[PoseInstance] = Field(default_factory=list)


class AnnotationOut(BaseModel):
    image: str
    boxes: list[BBox] = Field(default_factory=list)
    polygons: list[PolygonInstance] = Field(default_factory=list)
    poses: list[PoseInstance] = Field(default_factory=list)


class SamAssistRequest(BaseModel):
    """SAM2 点提示：归一化坐标。"""

    image: str = Field(..., min_length=1, description="图片文件名")
    x: float = Field(..., ge=0, le=1, description="归一化 X")
    y: float = Field(..., ge=0, le=1, description="归一化 Y")
    positive: bool = Field(default=True, description="正点击 True / 负点击 False")


class SamAssistOut(BaseModel):
    """SAM2 返回的归一化多边形。"""

    points: list[Point2D]
    model: str = "sam2_b.pt"
    width: int = 0
    height: int = 0


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
