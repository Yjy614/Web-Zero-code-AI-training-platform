"""运行时设置持久化（演示开关、大模型/视觉模型配置）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_ROOT, get_settings

RUNTIME_FILE = BACKEND_ROOT / "config" / "runtime_settings.json"

_DEFAULT_LLM = {
    "base_url": "",
    "api_key": "",
    "model": "",
    "timeout": 60,
    # DeepSeek 思考模式：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
    "thinking_enabled": True,
    "reasoning_effort": "high",
}

_DEFAULT_MENU_VISIBILITY = {
    "detect_wizard": True,
    "segment_wizard": True,
    "pose_wizard": True,
    "datasets": True,
    "models": True,
    "infer": True,
    "agent_chat": True,
    "agent_orchestrate": True,
    "weights": True,
    "settings": True,
    # 用户管理不在此列：仅管理员可见
}

_DEFAULT_VISION = {
    "base_url": "",
    "api_key": "",
    "model": "",
    "timeout": 60,
}


def _read_file() -> dict[str, Any]:
    if not RUNTIME_FILE.exists():
        return {}
    try:
        data = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_file(data: dict[str, Any]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_runtime_settings() -> dict[str, Any]:
    """合并 YAML 默认值与运行时覆盖。"""
    base = get_settings()
    raw = _read_file()
    menu_vis = {**_DEFAULT_MENU_VISIBILITY, **(raw.get("menu_visibility") or {})}
    # 只保留已知键，且强制为 bool
    menu_visibility = {k: bool(menu_vis.get(k, True)) for k in _DEFAULT_MENU_VISIBILITY}
    return {
        "demo_mode": raw.get("demo_mode", base.demo_mode),
        "job_runner": base.job_runner,
        "forbid_weight_download": base.forbid_weight_download,
        "free_step_nav": base.free_step_nav,
        "menu_visibility": menu_visibility,
        "llm": {**_DEFAULT_LLM, **(raw.get("llm") or {})},
        "vision": {**_DEFAULT_VISION, **(raw.get("vision") or {})},
    }


def update_runtime_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """更新并持久化运行时设置。"""
    raw = _read_file()
    if "demo_mode" in patch and patch["demo_mode"] is not None:
        raw["demo_mode"] = bool(patch["demo_mode"])
    if "menu_visibility" in patch and isinstance(patch["menu_visibility"], dict):
        cur = {**_DEFAULT_MENU_VISIBILITY, **(raw.get("menu_visibility") or {})}
        for k, v in patch["menu_visibility"].items():
            if k not in _DEFAULT_MENU_VISIBILITY or v is None:
                continue
            cur[k] = bool(v)
        raw["menu_visibility"] = {k: bool(cur.get(k, True)) for k in _DEFAULT_MENU_VISIBILITY}
    if "llm" in patch and isinstance(patch["llm"], dict):
        cur = {**_DEFAULT_LLM, **(raw.get("llm") or {})}
        for k, v in patch["llm"].items():
            if v is None:
                continue
            # 若前端传回掩码密钥则不覆盖原文
            if k == "api_key" and isinstance(v, str) and v.startswith("••••"):
                continue
            if k == "reasoning_effort":
                effort = str(v).strip().lower()
                cur[k] = effort if effort in {"low", "high", "max"} else "high"
                continue
            if k == "thinking_enabled":
                cur[k] = bool(v)
                continue
            cur[k] = v
        raw["llm"] = cur
    if "vision" in patch and isinstance(patch["vision"], dict):
        cur = {**_DEFAULT_VISION, **(raw.get("vision") or {})}
        for k, v in patch["vision"].items():
            if v is None:
                continue
            if k == "api_key" and isinstance(v, str) and v.startswith("••••"):
                continue
            cur[k] = v
        raw["vision"] = cur
    _write_file(raw)
    return get_runtime_settings()


def mask_secret(value: str) -> str:
    """对外展示时掩码 api_key。"""
    if not value:
        return ""
    if len(value) <= 4:
        return "••••"
    return "••••" + value[-4:]


def public_settings(*, mask_secrets: bool = True) -> dict[str, Any]:
    """返回给前端的设置（可掩码密钥）。"""
    data = get_runtime_settings()
    if mask_secrets:
        llm = dict(data["llm"])
        vision = dict(data["vision"])
        if llm.get("api_key"):
            llm["api_key"] = mask_secret(str(llm["api_key"]))
            llm["api_key_set"] = True
        else:
            llm["api_key_set"] = False
        if vision.get("api_key"):
            vision["api_key"] = mask_secret(str(vision["api_key"]))
            vision["api_key_set"] = True
        else:
            vision["api_key_set"] = False
        data["llm"] = llm
        data["vision"] = vision
    return data


def is_demo_mode() -> bool:
    return bool(get_runtime_settings().get("demo_mode", True))
