"""预训练权重解析：仅使用本地已上传文件，禁止联网下载。"""

from __future__ import annotations

from pathlib import Path

from app.services import task_storage


def normalize_weight_name(name: str) -> str:
    """规范化权重文件名；历史占位/乱码串返回空。"""
    import re

    raw = (name or "").strip()
    if not raw:
        return ""
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


def resolve_pretrained_weight(name: str, task_type: str = "detect") -> Path:
    """
    在 pretrained/<task_type>/ 下解析权重文件。
    不存在则抛出 FileNotFoundError（中文说明）。
    """
    fname = normalize_weight_name(name)
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
