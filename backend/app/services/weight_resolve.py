"""预训练权重解析：仅使用本地已上传文件，禁止联网下载。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.services import task_storage


def normalize_weight_name(name: str) -> str:
    """规范化权重文件名；历史占位/乱码串返回空。支持 @model:{id} 续训引用。"""
    import re

    raw = (name or "").strip()
    if not raw:
        return ""
    if raw.startswith("@model:"):
        try:
            int(raw.split(":", 1)[1])
        except ValueError:
            return ""
        return raw
    if "占位" in raw or "未上传" in raw:
        return ""
    m = re.search(r"[\w.-]+\.(?:pt|pth|onnx)", raw, flags=re.IGNORECASE)
    if not m:
        return ""
    fname = m.group(0)
    # 文件名后仍有多余字符（括号说明或编码乱码）→ 视为无效占位
    if raw != fname:
        return ""
    return Path(fname).name


def resolve_pretrained_weight(name: str, task_type: str = "detect", db: Session | None = None) -> Path:
    """
    解析训练起点权重：
    - 普通文件名 → pretrained/<task_type>/
    - @model:{id} → 模型库归档 PT（主动学习续训）
    """
    raw = (name or "").strip()
    if raw.startswith("@model:"):
        if db is None:
            raise FileNotFoundError("解析模型库权重需要数据库会话")
        try:
            mid = int(raw.split(":", 1)[1])
        except ValueError as e:
            raise FileNotFoundError(f"无效的模型权重引用：{raw}") from e
        from app.models.train import ModelRecord
        from app.services import model_artifacts

        row = db.query(ModelRecord).filter(ModelRecord.id == mid).first()
        if not row:
            raise FileNotFoundError(f"模型不存在：#{mid}")
        pt = model_artifacts.find_pt_path(row)
        if pt is None or not pt.is_file():
            raise FileNotFoundError(f"模型 #{mid} 没有可用的 PT 权重")
        return pt.resolve()

    fname = normalize_weight_name(raw)
    if not fname:
        raise FileNotFoundError("未选择预训练权重，请在配置步选择或上传 .pt 文件")
    tt = task_storage.normalize_weight_task_type(task_type)
    root = task_storage.weights_dir(tt)
    root.mkdir(parents=True, exist_ok=True)
    path = root / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"[{tt}] 预训练权重不存在：{fname}。请管理员在「权重仓库」对应任务类型下上传（禁止自动下载）。"
        )
    return path.resolve()
