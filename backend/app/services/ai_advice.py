"""评估后 AI 优化建议：结合数据集规模与可调超参。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.dataset import Dataset
from app.models.train import TrainTask
from app.services import dataset_storage
from app.services.llm_client import chat_completion
from app.services.runtime_settings import get_runtime_settings, is_demo_mode


def _parse_classes(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return []


def _count_labeled(root: Path) -> int:
    """统计已有标注文件的图片数。"""
    if not root.exists():
        return 0
    n = 0
    for name in dataset_storage.list_images(root):
        if dataset_storage.label_path_for(root, name).exists():
            n += 1
    return n


def build_advice_context(task: TrainTask, dataset: Dataset) -> dict[str, Any]:
    """汇总用户可调整的上下文（数据量 + 训练参数 + 评估指标）。"""
    classes = _parse_classes(dataset.classes_json)
    root = Path(dataset.path) if dataset.path else None
    labeled = _count_labeled(root) if root else 0
    image_count = int(dataset.image_count or 0)
    try:
        config = json.loads(task.config_json or "{}")
        if not isinstance(config, dict):
            config = {}
    except json.JSONDecodeError:
        config = {}
    try:
        metrics = json.loads(task.metrics_json or "{}")
        if not isinstance(metrics, dict):
            metrics = {}
    except json.JSONDecodeError:
        metrics = {}

    # 训练曲线历史可能很大，只取末尾摘要
    history = metrics.get("history")
    history_tail: list[Any] = []
    if isinstance(history, list) and history:
        history_tail = history[-3:]

    return {
        "task_name": task.name,
        "task_status": task.status,
        "demo_mode": is_demo_mode(),
        "dataset": {
            "name": dataset.name,
            "image_count": image_count,
            "labeled_count": labeled,
            "label_ratio": round(labeled / image_count, 3) if image_count else 0,
            "class_count": len(classes),
            "classes": classes[:30],
        },
        "train_config": {
            "train_ratio": config.get("train_ratio"),
            "val_ratio": config.get("val_ratio"),
            "epochs": config.get("epochs"),
            "batch": config.get("batch"),
            "imgsz": config.get("imgsz"),
            "train_strategy": config.get("train_strategy"),
            "device": config.get("device"),
            "pretrained_weight": config.get("pretrained_weight"),
            "augment": config.get("augment"),
        },
        "eval_metrics": {
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "history_tail": history_tail,
        },
    }


SYSTEM_PROMPT = """你是工业场景目标检测（YOLO）训练顾问。用户使用零代码平台，只能调整：
1) 数据集：图片数量、类别、标注完整度与质量、难例补充；
2) 训练参数：train/val/test 划分比例、epochs、batch、imgsz、训练策略（稳健/快速收敛/细调）、是否增强、预训练权重选择、设备。

请根据给出的 JSON 上下文，用简体中文给出可执行的优化建议。
格式要求（必须严格遵守）：
- 禁止使用任何 Markdown：不要使用 **加粗**、# 标题、-/* 列表符号、反引号等；
- 先写一行「数据侧」，再写若干条建议；然后写一行「参数侧」，再写若干条建议；
- 每条建议单独一行，格式为：星星 + 空格 + 序号. + 内容
  示例：⭐⭐⭐ 1. 图片仅 5 张，建议补充至数百张并覆盖光照变化。
- 用 ⭐ 数量表示优先级：⭐⭐⭐ 最高，⭐⭐ 中，⭐ 较低；每条 1～3 颗星；
- 每条说明原因与建议操作；不要编造未提供的指标；
- 总长度控制在 400 字以内；不要建议下载公网权重或改服务器代码。"""


def sanitize_advice_text(text: str) -> str:
    """去掉模型误输出的 Markdown 标记，保留 ⭐ 与正文。"""
    import re

    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""
    # 去掉加粗/斜体星号（保留单独的 ⭐ 字符）
    raw = raw.replace("**", "").replace("__", "")
    raw = re.sub(r"(?<!\*)\*(?!\*)", "", raw)
    # 去掉标题井号
    raw = re.sub(r"(?m)^#{1,6}\s*", "", raw)
    # 去掉行首无序列表符 -
    raw = re.sub(r"(?m)^\s*[-•]\s+", "", raw)
    # 压缩多余空行
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _fallback_advice(ctx: dict[str, Any]) -> str:
    """未配置或调用失败时的规则建议。"""
    ds = ctx.get("dataset") or {}
    cfg = ctx.get("train_config") or {}
    met = ctx.get("eval_metrics") or {}
    data_tips: list[tuple[str, str]] = []
    param_tips: list[tuple[str, str]] = []

    img_n = int(ds.get("image_count") or 0)
    labeled = int(ds.get("labeled_count") or 0)
    ratio = float(ds.get("label_ratio") or 0)
    classes_n = int(ds.get("class_count") or 0)

    if img_n < 50:
        data_tips.append(
            ("⭐⭐⭐", f"当前图片约 {img_n} 张，偏少。建议补充同工况样本至数百张以上，并覆盖光照/角度变化。")
        )
    elif img_n < 200:
        data_tips.append(("⭐⭐", f"图片约 {img_n} 张，可继续增加难例与边界场景，提升泛化。"))

    if img_n and ratio < 0.9:
        data_tips.append(
            ("⭐⭐⭐", f"已标注 {labeled}/{img_n}（约 {ratio:.0%}），请先补全标注再训练，避免漏标拖低召回。")
        )
    if classes_n <= 0:
        data_tips.append(("⭐⭐⭐", "尚未配置类别，请在标注步添加类别并完成框选。"))

    epochs = int(cfg.get("epochs") or 0)
    batch = int(cfg.get("batch") or 0)
    imgsz = int(cfg.get("imgsz") or 0)
    train_ratio = float(cfg.get("train_ratio") or 0)
    augment = cfg.get("augment")

    if epochs and epochs < 30:
        param_tips.append(("⭐⭐", f"epochs={epochs} 偏短，可先试 50–100，观察验证 mAP 是否仍上升。"))
    elif epochs and epochs > 200 and img_n < 100:
        param_tips.append(("⭐⭐", f"数据量不大但 epochs={epochs} 很高，易过拟合，可降低轮数或加强增强。"))

    if batch and batch > 16 and img_n < 100:
        param_tips.append(("⭐", f"batch={batch} 较大而样本少，可降到 4–8 提高迭代次数。"))
    if imgsz and imgsz < 640:
        param_tips.append(("⭐", f"imgsz={imgsz}，若目标偏小可试 640/960；若显存不足再降回。"))

    strategy = str(cfg.get("train_strategy") or "stable")
    if strategy == "fast" and img_n < 50:
        param_tips.append(
            ("⭐", "样本较少仍用「快速收敛」时曲线可能抖动，可改用「稳健」或「细调」。")
        )
    elif strategy == "finetune" and epochs and epochs < 40:
        param_tips.append(
            ("⭐", "「细调」学习率较小，epochs 偏短时提升有限，可适当增加轮数或改用「稳健」。")
        )

    if train_ratio and train_ratio > 0.9:
        param_tips.append(("⭐⭐", f"训练集比例 {train_ratio} 过高，验证集过小，建议留出约 15%–20% 做验证。"))
    if augment is False:
        param_tips.append(("⭐⭐", "当前关闭数据增强，小样本场景建议开启增强以缓解过拟合。"))

    map50 = met.get("map50")
    recall = met.get("recall")
    precision = met.get("precision")
    try:
        if map50 is not None and float(map50) < 0.5:
            data_tips.append(("⭐⭐⭐", "mAP@0.5 偏低：优先检查漏标/错标，再考虑增加同类样本或提高 epochs。"))
        if recall is not None and precision is not None and float(recall) + 0.15 < float(precision):
            data_tips.append(("⭐⭐", "召回明显低于精确率：可能漏检，建议补难例、适当增强或检查框是否过严。"))
        if recall is not None and precision is not None and float(precision) + 0.15 < float(recall):
            data_tips.append(("⭐⭐", "精确率偏低：可能误检，检查负样本/背景干扰，或收紧标注一致性。"))
    except (TypeError, ValueError):
        pass

    if not data_tips and not param_tips:
        data_tips.append(("⭐", "指标与数据规模尚可。可从难例挖掘、标注一致性抽检做小步迭代。"))
        param_tips.append(("⭐", "可微调 epochs / imgsz 观察验证集变化。"))

    lines = ["规则建议（未调用大模型或调用失败时的本地建议）", "", "数据侧"]
    for i, (stars, text) in enumerate(data_tips, 1):
        lines.append(f"{stars} {i}. {text}")
    lines.extend(["", "参数侧"])
    for i, (stars, text) in enumerate(param_tips, 1):
        lines.append(f"{stars} {i}. {text}")
    return "\n".join(lines)


def generate_ai_advice(task: TrainTask, dataset: Dataset) -> dict[str, Any]:
    """
    生成 AI 优化建议。
    返回：advice / source(llm|fallback) / context / error(可选)
    """
    ctx = build_advice_context(task, dataset)
    settings = get_runtime_settings()
    llm = settings.get("llm") or {}
    base_url = str(llm.get("base_url") or "").strip()
    api_key = str(llm.get("api_key") or "").strip()
    model = str(llm.get("model") or "").strip()
    timeout = int(llm.get("timeout") or 60)

    if not base_url or not model:
        return {
            "advice": sanitize_advice_text(
                _fallback_advice(ctx)
                + "\n\n提示：请在「设置」中配置大模型 base_url 与 model 后重新获取 AI 建议。"
            ),
            "source": "fallback",
            "context": ctx,
            "error": "llm_not_configured",
        }

    user_content = (
        "以下是当前检测任务的上下文 JSON，请给出优化建议：\n"
        + json.dumps(ctx, ensure_ascii=False, indent=2)
    )
    try:
        text = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return {"advice": sanitize_advice_text(text), "source": "llm", "context": ctx}
    except Exception as e:  # noqa: BLE001 — 降级为规则建议
        return {
            "advice": sanitize_advice_text(_fallback_advice(ctx) + f"\n\n（大模型调用失败：{e}）"),
            "source": "fallback",
            "context": ctx,
            "error": str(e),
        }
