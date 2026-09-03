"""系统信息与设置。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_current_user, require_admin
from app.core.config import get_settings
from app.models.user import User
from app.services import runtime_settings

router = APIRouter(tags=["系统"])


class SystemInfo(BaseModel):
    api_version: str
    demo_mode: bool
    job_runner: str
    storage_configured: bool = True
    forbid_weight_download: bool


class DeviceItem(BaseModel):
    id: str
    label: str


class DevicesOut(BaseModel):
    devices: list[DeviceItem]
    cuda_available: bool = False
    gpu_count: int = 0


class ModelEndpointConfig(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = Field(default=60, ge=5, le=600)
    api_key_set: bool = False
    # DeepSeek 思考模式（仅 LLM 使用；vision 忽略）
    thinking_enabled: bool = True
    reasoning_effort: str = Field(default="high", description="low | high | max")


class MenuVisibility(BaseModel):
    detect_wizard: bool = True
    segment_wizard: bool = True
    pose_wizard: bool = True
    datasets: bool = True
    models: bool = True
    infer: bool = True
    agent_chat: bool = True
    agent_orchestrate: bool = True
    weights: bool = True
    settings: bool = True


class SettingsOut(BaseModel):
    demo_mode: bool
    job_runner: str
    forbid_weight_download: bool
    free_step_nav: bool
    menu_visibility: MenuVisibility
    llm: ModelEndpointConfig
    vision: ModelEndpointConfig


class ModelEndpointUpdate(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: int | None = Field(default=None, ge=5, le=600)
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None


class SettingsUpdate(BaseModel):
    demo_mode: bool | None = None
    llm: ModelEndpointUpdate | None = None
    vision: ModelEndpointUpdate | None = None
    menu_visibility: MenuVisibility | None = None


def _to_settings_out() -> SettingsOut:
    data = runtime_settings.public_settings(mask_secrets=True)
    llm = dict(data["llm"] or {})
    vision = dict(data["vision"] or {})
    # vision 不使用思考模式字段，给默认值避免校验失败
    llm.setdefault("thinking_enabled", True)
    llm.setdefault("reasoning_effort", "high")
    vision.setdefault("thinking_enabled", False)
    vision.setdefault("reasoning_effort", "high")
    return SettingsOut(
        demo_mode=data["demo_mode"],
        job_runner=data["job_runner"],
        forbid_weight_download=data["forbid_weight_download"],
        free_step_nav=data["free_step_nav"],
        menu_visibility=MenuVisibility(**(data.get("menu_visibility") or {})),
        llm=ModelEndpointConfig(**llm),
        vision=ModelEndpointConfig(**vision),
    )


@router.get("/system/info", response_model=SystemInfo)
def system_info(_: User = Depends(get_current_user)) -> SystemInfo:
    """本地环境与演示状态（不暴露服务器绝对路径）。"""
    s = get_settings()
    rt = runtime_settings.get_runtime_settings()
    return SystemInfo(
        api_version=s.api_version,
        demo_mode=bool(rt["demo_mode"]),
        job_runner=s.job_runner,
        storage_configured=True,
        forbid_weight_download=s.forbid_weight_download,
    )


@router.get("/system/devices", response_model=DevicesOut)
def list_compute_devices(_: User = Depends(get_current_user)) -> DevicesOut:
    """探测本机可用训练设备（CPU / CUDA GPU），供配置步下拉使用。"""
    devices: list[DeviceItem] = [DeviceItem(id="cpu", label="CPU")]
    cuda_available = False
    gpu_count = 0
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            gpu_count = int(torch.cuda.device_count())
            for i in range(gpu_count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:  # noqa: BLE001
                    name = f"GPU {i}"
                devices.append(DeviceItem(id=f"cuda:{i}", label=f"cuda:{i} · {name}"))
    except ImportError:
        # 未安装 torch 时仅提供 CPU
        pass
    return DevicesOut(devices=devices, cuda_available=cuda_available, gpu_count=gpu_count)


@router.get("/settings", response_model=SettingsOut)
def get_app_settings(_: User = Depends(get_current_user)) -> SettingsOut:
    """读取设置（密钥已掩码）。"""
    return _to_settings_out()


@router.put("/settings", response_model=SettingsOut)
def put_app_settings(body: SettingsUpdate, _: User = Depends(require_admin)) -> SettingsOut:
    """更新设置（仅管理员）。"""
    patch: dict[str, Any] = {}
    if body.demo_mode is not None:
        patch["demo_mode"] = body.demo_mode
    if body.llm is not None:
        patch["llm"] = body.llm.model_dump(exclude_none=True)
    if body.vision is not None:
        patch["vision"] = body.vision.model_dump(exclude_none=True)
    if body.menu_visibility is not None:
        patch["menu_visibility"] = body.menu_visibility.model_dump()
    runtime_settings.update_runtime_settings(patch)
    return _to_settings_out()
