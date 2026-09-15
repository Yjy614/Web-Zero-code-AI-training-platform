"""主动学习会话持久化（文件 JSON，任务无关）。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import storage_root_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sessions_root(user_id: int) -> Path:
    root = storage_root_path() / "active_learn" / f"user_{int(user_id)}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_dir(user_id: int, session_id: str) -> Path:
    d = sessions_root(user_id) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_path(user_id: int, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "session.json"


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def create_session(user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    sid = new_session_id()
    data = {
        "id": sid,
        "user_id": int(user_id),
        "status": "created",  # created|uploaded|screening|screened|merged|retraining|done|error
        "created_at": _now(),
        "updated_at": _now(),
        "model_id": payload.get("model_id"),
        "model_name": payload.get("model_name"),
        "task_type": payload.get("task_type") or "detect",
        "target_dataset_id": payload.get("target_dataset_id"),
        "target_dataset_name": payload.get("target_dataset_name"),
        "staging_dataset_id": payload.get("staging_dataset_id"),
        "staging_dataset_name": payload.get("staging_dataset_name"),
        "job_id": None,
        "retrain_task_id": None,
        "retrain_job_id": None,
        "summary": {},
        "items": [],  # list of ImageScore dicts
        "error": None,
    }
    save_session(user_id, data)
    return data


def load_session(user_id: int, session_id: str) -> dict[str, Any]:
    path = session_path(user_id, session_id)
    if not path.is_file():
        raise FileNotFoundError(f"主动学习会话不存在：{session_id}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("会话文件损坏")
    if int(raw.get("user_id") or 0) != int(user_id):
        raise PermissionError("无权访问该会话")
    return raw


def save_session(user_id: int, data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    data["updated_at"] = _now()
    path = session_path(user_id, str(data["id"]))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def list_sessions(user_id: int, *, limit: int = 30) -> list[dict[str, Any]]:
    root = sessions_root(user_id)
    rows: list[dict[str, Any]] = []
    for p in sorted(root.glob("*/session.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                rows.append(raw)
        except Exception:  # noqa: BLE001
            continue
        if len(rows) >= limit:
            break
    return rows
