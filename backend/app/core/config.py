"""应用配置：读取 YAML 与环境变量。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 根目录
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_ROOT / "config" / "web_config.yaml"


class Settings(BaseSettings):
    """运行时配置。"""

    model_config = SettingsConfigDict(env_prefix="AI_PLATFORM_", extra="ignore")

    demo_mode: bool = True
    storage_root: str = "./storage"
    # 管理员上传的全局预训练权重（与用户 storage 分离）
    pretrained_root: str = "./pretrained"
    job_runner: str = "mock"
    forbid_weight_download: bool = True
    jwt_secret: str = "change-me-in-production-ai-platform"
    jwt_expire_minutes: int = 720
    database_url: str = "sqlite:///./platform_web.db"
    free_step_nav: bool = True
    seed_users: list[dict[str, Any]] = Field(default_factory=list)
    api_version: str = "0.1.0"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


@lru_cache
def get_settings() -> Settings:
    """加载配置（YAML 优先，环境变量可覆盖）。"""
    raw = _load_yaml(DEFAULT_CONFIG_PATH)
    debug = raw.pop("debug", {}) or {}
    if isinstance(debug, dict) and "free_step_nav" in debug:
        raw["free_step_nav"] = debug["free_step_nav"]
    return Settings(**raw)


def storage_root_path() -> Path:
    """解析用户数据存储根目录为绝对路径。"""
    root = Path(get_settings().storage_root)
    if not root.is_absolute():
        root = (BACKEND_ROOT / root).resolve()
    return root


def pretrained_root_path() -> Path:
    """解析系统预训练权重根目录（管理员维护，与 storage 分离）。"""
    root = Path(get_settings().pretrained_root)
    if not root.is_absolute():
        root = (BACKEND_ROOT / root).resolve()
    return root
