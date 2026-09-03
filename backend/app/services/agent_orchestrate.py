"""
AI 流程编排：自然语言 → 步骤计划 → 逐步确认执行。

标注策略：不根据 NL 直接写标签；可选预标注 + 强制人工抽检门禁。
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.core.config import storage_root_path
from app.core.task_types import TASK_TYPE_LABELS, normalize_task_type
from app.models.dataset import Dataset
from app.models.train import Job, TrainTask
from app.models.user import User
from app.schemas.agent import AgentPlanOut, AgentPlanRequest, AgentStepOut
from app.schemas.task import TrainConfig
from app.services import dataset_prelabel, dataset_storage, task_storage
from app.services.bootstrap import username_by_id
from app.services.llm_client import chat_completion
from app.services.runtime_settings import get_runtime_settings
from app.services.weight_resolve import normalize_weight_name, resolve_pretrained_weight


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plans_dir(owner_id: int) -> Path:
    d = storage_root_path() / "agent_plans" / f"user_{owner_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plan_path(owner_id: int, plan_id: str) -> Path:
    return _plans_dir(owner_id) / f"{plan_id}.json"


def save_plan(plan: AgentPlanOut) -> None:
    plan.updated_at = _now_iso()
    path = _plan_path(plan.owner_id, plan.id)
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan(owner_id: int, plan_id: str) -> AgentPlanOut | None:
    path = _plan_path(owner_id, plan_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentPlanOut.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def _step(
    *,
    sid: str,
    typ: str,
    title: str,
    description: str,
    human_gate: bool = False,
    params: dict[str, Any] | None = None,
) -> AgentStepOut:
    return AgentStepOut(
        id=sid,
        type=typ,  # type: ignore[arg-type]
        title=title,
        description=description,
        human_gate=human_gate,
        status="pending",
        params=params or {},
    )


def _default_weight_name(task_type: str) -> str:
    tt = normalize_task_type(task_type)
    if tt == "segment":
        preferred = ("yolov8n-seg.pt", "yolo11n-seg.pt", "yolov8s-seg.pt")
    elif tt == "pose":
        preferred = ("yolo11n-pose.pt", "yolov8n-pose.pt", "yolo11s-pose.pt", "yolov8s-pose.pt")
    else:
        preferred = ("yolov8n.pt", "yolo11n.pt", "yolov8s.pt")
    for name in preferred:
        try:
            resolve_pretrained_weight(name, tt)
            return name
        except FileNotFoundError:
            continue
    files = task_storage.list_weight_files(tt)
    for item in files:
        name = str(item.get("name") or "")
        if name.lower().endswith(".pt"):
            return name
    return preferred[0]


def _normalize_train_device(device: str) -> str:
    """将设备字符串规范化为本机可用 id（cpu / cuda:N）。无 CUDA 时回退 cpu。"""
    raw = (device or "cpu").strip().lower()
    if raw in {"", "cpu"}:
        return "cpu"
    # 兼容旧值 "0" / "gpu" / "cuda"
    if raw in {"gpu", "cuda"}:
        raw = "cuda:0"
    if raw.isdigit():
        raw = f"cuda:{raw}"
    allowed = {"cpu"}
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(int(torch.cuda.device_count())):
                allowed.add(f"cuda:{i}")
    except ImportError:
        pass
    if raw in allowed:
        return raw
    return "cpu"


def parse_prompt_intent(prompt: str, *, task_type: str = "detect") -> dict[str, Any]:
    """
    从自然语言需求解析 epochs / 开关 / 权重 / batch / device。
    未提及的项用合理默认。
    """
    text = (prompt or "").strip()
    tt = normalize_task_type(task_type)
    hints: list[str] = []

    epochs = 50
    m = re.search(r"(\d+)\s*轮", text)
    if not m:
        m = re.search(r"(\d+)\s*epochs?", text, re.IGNORECASE)
    if not m:
        m = re.search(r"训练\s*(\d+)", text)
    if m:
        epochs = max(1, min(500, int(m.group(1))))

    include_prelabel = bool(re.search(r"预标注|pre-?\s*label", text, re.IGNORECASE))

    include_eval = True
    if re.search(r"不(做|要|进行)?评估|跳过评估|无需评估|不要\s*eval", text, re.IGNORECASE):
        include_eval = False
    elif re.search(r"评估|eval", text, re.IGNORECASE):
        include_eval = True

    include_export = True
    if re.search(r"不(做|要|进行)?导出|跳过导出|无需导出", text):
        include_export = False
    elif re.search(r"导出|onnx|ONNX", text):
        include_export = True

    batch = 8
    bm = re.search(r"batch\s*[=:：]?\s*(\d+)", text, re.IGNORECASE)
    if not bm:
        bm = re.search(r"批次\s*[=:：]?\s*(\d+)", text)
    if bm:
        batch = max(1, min(128, int(bm.group(1))))

    device = "cpu"
    want_gpu = bool(re.search(r"\bcuda\b|\bgpu\b|显卡|0\s*号卡", text, re.IGNORECASE))
    want_cpu = bool(re.search(r"\bcpu\b", text, re.IGNORECASE))
    if want_cpu and not want_gpu:
        device = "cpu"
    elif want_gpu:
        # 与 /system/devices 返回的 id 对齐：有 CUDA 时用 cuda:0
        try:
            import torch

            if torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
                device = "cuda:0"
            else:
                device = "cpu"
                hints.append("需求提到 GPU，但本机未检测到可用 CUDA，已回退为 CPU")
        except ImportError:
            device = "cpu"
            hints.append("需求提到 GPU，但当前环境无 torch/CUDA，已回退为 CPU")

    weight: str | None = None
    wm = re.search(r"(yolo[\w.-]+\.pt)", text, re.IGNORECASE)
    if not wm:
        wm = re.search(
            r"\b(yolo(?:v8|11)[nsmxl](?:-pose|-seg)?)\b",
            text,
            re.IGNORECASE,
        )
    if wm:
        raw = wm.group(1)
        weight = raw if raw.lower().endswith(".pt") else f"{raw}.pt"
        weight = normalize_weight_name(weight)
        # 与任务类型冲突提示
        low = weight.lower()
        if tt == "pose" and "-pose" not in low:
            hints.append(f"需求写了权重「{weight}」，但当前是姿态任务，建议使用 *-pose.pt")
        elif tt == "segment" and "-seg" not in low:
            hints.append(f"需求写了权重「{weight}」，但当前是分割任务，建议使用 *-seg.pt")
        elif tt == "detect" and ("-pose" in low or "-seg" in low):
            hints.append(f"需求写了权重「{weight}」，与目标检测任务可能不匹配")
        try:
            resolve_pretrained_weight(weight, tt)
        except FileNotFoundError:
            hints.append(f"权重仓库中未找到「{weight}」，将回退默认权重；请到权重仓库上传")
            weight = None

    return {
        "epochs": epochs,
        "include_eval": include_eval,
        "include_export": include_export,
        "include_prelabel": include_prelabel,
        "batch": batch,
        "device": device,
        "pretrained_weight": weight,
        "hints": hints,
    }


def build_template_plan(req: AgentPlanRequest, *, owner_id: int, dataset_name: str) -> AgentPlanOut:
    """无 LLM 或解析失败时的固定模板计划。"""
    tt = normalize_task_type(req.task_type)
    intent = parse_prompt_intent(req.prompt, task_type=tt)
    # 以需求描述解析为准；显式传 epochs 时仍可覆盖（兼容旧调用）
    epochs = int(req.epochs) if req.epochs is not None else int(intent["epochs"])
    include_prelabel = bool(intent["include_prelabel"])
    include_eval = bool(intent["include_eval"])
    include_export = bool(intent["include_export"])
    batch = int(intent.get("batch") or 8)
    device = str(intent.get("device") or "cpu")
    device = _normalize_train_device(device)
    weight = str(intent.get("pretrained_weight") or "") or _default_weight_name(tt)
    hints = list(intent.get("hints") or [])
    if not task_storage.list_weight_files(tt):
        hints.append(f"当前无可用的 {TASK_TYPE_LABELS.get(tt, tt)} 预训练权重，请先上传到权重仓库")
    steps: list[AgentStepOut] = [
        _step(
            sid="s1",
            typ="ensure_task",
            title="准备训练任务",
            description=f"绑定数据集「{dataset_name}」，创建或复用训练任务。",
            params={"task_name": f"{dataset_name}_agent"},
        ),
        _step(
            sid="s2",
            typ="check_labels",
            title="检查标注门槛",
            description="检查已标注数量是否达到训练/预标注门槛；不足则停止并提示先去标注。",
        ),
    ]
    if include_prelabel:
        steps.append(
            _step(
                sid="s2b",
                typ="prelabel",
                title="AI 预标注（可选）",
                description="用已标注图短训后补全未标注图；写入后可撤销。需人工抽检。",
                human_gate=True,
            )
        )
    steps.extend(
        [
            _step(
                sid="s3",
                typ="review_labels",
                title="人工抽检标注",
                description="请打开标注页抽检预标注/人工标注质量。确认无误后再继续训练。",
                human_gate=True,
            ),
            _step(
                sid="s4",
                typ="apply_config",
                title="写入训练配置",
                description=f"epochs={epochs}，batch={batch}，device={device}，权重={weight}，策略=stable。",
                params={
                    "epochs": epochs,
                    "batch": batch,
                    "imgsz": 640,
                    "train_strategy": "stable",
                    "device": device,
                    "pretrained_weight": weight,
                    "train_ratio": 0.7,
                    "val_ratio": 0.2,
                    "test_ratio": 0.1,
                    "augment": True,
                },
            ),
            _step(
                sid="s5",
                typ="split",
                title="划分数据集",
                description="按配置比例生成 train/val/test。",
            ),
            _step(
                sid="s6",
                typ="train",
                title="开始训练",
                description="提交训练 Job，完成后自动进入下一步。",
                human_gate=True,
            ),
        ]
    )
    if include_eval:
        steps.append(
            _step(
                sid="s7",
                typ="eval",
                title="评估",
                description="对训练产物做评估并生成报告。",
                human_gate=True,
            )
        )
    if include_export:
        steps.append(
            _step(
                sid="s8",
                typ="export",
                title="导出模型",
                description="导出 PT / ONNX 到版本化目录。",
                human_gate=True,
                params={"formats": ["pt", "onnx"]},
            )
        )
    # 不再追加「完成摘要」空步骤；最后一步结束后由前端展示结果落地页
    # 第一步 ready
    if steps:
        steps[0].status = "ready"

    kind = TASK_TYPE_LABELS.get(tt, tt)
    return AgentPlanOut(
        id=uuid.uuid4().hex[:12],
        owner_id=owner_id,
        prompt=req.prompt.strip(),
        task_type=tt,
        dataset_id=req.dataset_id,
        task_id=req.task_id,
        summary=(
            f"将按确认逐步执行：{kind} · 数据集「{dataset_name}」· {epochs} epochs · "
            f"评估={'开' if include_eval else '关'} · 导出={'开' if include_export else '关'}"
            + (" · 预标注开" if include_prelabel else "")
            + f" · 权重 {weight}"
        ),
        source="template",
        status="draft",
        current_index=0,
        steps=steps,
        hints=hints,
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    # 允许 ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _merge_llm_overrides(plan: AgentPlanOut, llm_obj: dict[str, Any]) -> AgentPlanOut:
    """用 LLM 解析出的 epochs / 开关覆盖模板。"""
    epochs = llm_obj.get("epochs")
    if isinstance(epochs, int) and 1 <= epochs <= 500:
        for s in plan.steps:
            if s.type == "apply_config":
                s.params["epochs"] = epochs
                s.description = f"epochs={epochs}，权重={s.params.get('pretrained_weight')}，策略=stable。"
    summary = llm_obj.get("summary")
    if isinstance(summary, str) and summary.strip():
        plan.summary = summary.strip()[:300]
    # 开关：若 LLM 明确说不要 eval/export，从计划移除（仅在尚未执行时）
    if llm_obj.get("include_eval") is False:
        plan.steps = [s for s in plan.steps if s.type != "eval"]
    if llm_obj.get("include_export") is False:
        plan.steps = [s for s in plan.steps if s.type != "export"]
    if llm_obj.get("include_prelabel") is True and not any(s.type == "prelabel" for s in plan.steps):
        # 插入到 check_labels 之后
        idx = next((i for i, s in enumerate(plan.steps) if s.type == "check_labels"), 0) + 1
        plan.steps.insert(
            idx,
            _step(
                sid="s2b",
                typ="prelabel",
                title="AI 预标注（可选）",
                description="用已标注图短训后补全未标注图；写入后可撤销。需人工抽检。",
                human_gate=True,
            ),
        )
    # 重建 ready
    for i, s in enumerate(plan.steps):
        s.status = "ready" if i == 0 else "pending"
    plan.current_index = 0
    plan.source = "llm"
    return plan


def generate_plan(
    db: Session,
    user: User,
    req: AgentPlanRequest,
    dataset: Dataset,
) -> AgentPlanOut:
    """生成计划：优先 LLM 解析意图，失败则纯模板。"""
    plan = build_template_plan(req, owner_id=user.id, dataset_name=dataset.name)
    settings = get_runtime_settings()
    llm = settings.get("llm") or {}
    base_url = str(llm.get("base_url") or "").strip()
    api_key = str(llm.get("api_key") or "").strip()
    model = str(llm.get("model") or "").strip()
    timeout = int(llm.get("timeout") or 60)

    if not (base_url and model):
        save_plan(plan)
        return plan

    system = (
        "你是工业零代码 AI 训练平台的流程编排助手。"
        "根据用户意图，只输出一个 JSON 对象，不要其它文字。"
        "字段：summary(string), epochs(int|null), include_eval(bool), include_export(bool), "
        "include_prelabel(bool)。"
        "注意：不要根据自然语言直接生成标注坐标；标注只能人工或预标注+抽检。"
    )
    user_msg = (
        f"用户需求：{req.prompt}\n"
        f"任务类型：{plan.task_type}\n"
        f"数据集：{dataset.name} (id={dataset.id})\n"
        "请从用户需求中推断 epochs、是否评估/导出/预标注；未提及则 epochs=50、评估与导出为 true、预标注为 false。"
    )
    try:
        text = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            timeout=timeout,
            temperature=0.2,
        )
        obj = _extract_json_object(text)
        if obj:
            plan = _merge_llm_overrides(plan, obj)
    except Exception:  # noqa: BLE001
        plan.source = "template"

    save_plan(plan)
    return plan


def _mark_ready(plan: AgentPlanOut) -> None:
    """根据 current_index 刷新 ready/pending。"""
    for i, s in enumerate(plan.steps):
        if s.status in {"completed", "skipped", "failed"}:
            continue
        if i == plan.current_index:
            s.status = "ready"
        elif s.status == "ready":
            s.status = "pending"


def _advance(plan: AgentPlanOut) -> None:
    plan.current_index += 1
    while plan.current_index < len(plan.steps) and plan.steps[plan.current_index].status in {
        "completed",
        "skipped",
    }:
        plan.current_index += 1
    if plan.current_index >= len(plan.steps):
        plan.status = "completed"
        plan.current_index = max(0, len(plan.steps) - 1)
        # 用结果摘要覆盖计划头，供前端落地页展示
        tid = plan.task_id or "—"
        plan.summary = (
            f"编排已完成 · 任务类型 {TASK_TYPE_LABELS.get(plan.task_type, plan.task_type)} · "
            f"训练任务 #{tid}。可前往模型库查看产物，或打开推理试用。"
        )
    else:
        plan.status = "running"
        _mark_ready(plan)


async def run_current_step(
    *,
    db: Session,
    user: User,
    plan: AgentPlanOut,
    action: str,
    submit_job: Callable[[str, dict[str, Any]], Any],
) -> tuple[AgentPlanOut, str]:
    """
    执行当前步骤。submit_job(job_type, payload) 用于异步任务。
    返回 (plan, message)。
    """
    if plan.status in {"completed", "cancelled"}:
        raise ValueError("计划已结束，无法继续执行")
    if plan.current_index < 0 or plan.current_index >= len(plan.steps):
        raise ValueError("没有可执行的步骤")

    step = plan.steps[plan.current_index]
    if step.status not in {"ready", "pending", "failed"}:
        if step.status == "running" and step.job_id:
            raise ValueError("当前步骤仍有进行中的 Job，请等待完成后再继续")
        raise ValueError(f"当前步骤状态为 {step.status}，无法执行")

    if action == "skip":
        if step.type in {"check_labels", "ensure_task", "summary"}:
            raise ValueError("该步骤不可跳过")
        step.status = "skipped"
        step.result = {"skipped": True}
        _advance(plan)
        save_plan(plan)
        return plan, f"已跳过：{step.title}"

    step.status = "running"
    step.error = None
    plan.status = "running"
    save_plan(plan)

    try:
        msg = await _execute_step(db, user, plan, step, submit_job)
        # 异步 Job：保持 running，等 ack
        if step.job_id and step.status == "running":
            save_plan(plan)
            return plan, msg
        step.status = "completed"
        _advance(plan)
        save_plan(plan)
        return plan, msg
    except Exception as e:  # noqa: BLE001
        step.status = "failed"
        step.error = str(e)
        plan.status = "failed"
        save_plan(plan)
        raise


async def _execute_step(
    db: Session,
    user: User,
    plan: AgentPlanOut,
    step: AgentStepOut,
    submit_job: Callable[[str, dict[str, Any]], Any],
) -> str:
    ds = db.query(Dataset).filter(Dataset.id == plan.dataset_id).first()
    if not ds:
        raise ValueError("数据集不存在")

    if step.type == "ensure_task":
        task = None
        if plan.task_id:
            task = db.query(TrainTask).filter(TrainTask.id == plan.task_id).first()
            if task and user.role != "admin" and task.owner_id != user.id:
                raise ValueError("无权使用该训练任务")
        if not task:
            name = str(step.params.get("task_name") or f"{ds.name}_agent")
            # 同名则追加后缀
            base = name
            n = 1
            while (
                db.query(TrainTask)
                .filter(TrainTask.owner_id == user.id, TrainTask.name == name)
                .first()
            ):
                n += 1
                name = f"{base}_{n}"
            task = TrainTask(
                name=name,
                owner_id=user.id,
                dataset_id=ds.id,
                status="draft",
                step=3,
                config_json="{}",
            )
            db.add(task)
            db.commit()
            db.refresh(task)
        plan.task_id = task.id
        step.result = {"task_id": task.id, "task_name": task.name}
        return f"已准备训练任务「{task.name}」(#{task.id})"

    if not plan.task_id:
        raise ValueError("尚未绑定训练任务")
    task = db.query(TrainTask).filter(TrainTask.id == plan.task_id).first()
    if not task:
        raise ValueError("训练任务不存在")

    if step.type == "check_labels":
        root = Path(ds.path)
        info = dataset_prelabel.analyze_prelabel(root, plan.task_type)
        step.result = {
            "labeled_count": info["labeled_count"],
            "unlabeled_count": info["unlabeled_count"],
            "min_labeled": info["min_labeled"],
            "class_count": info["class_count"],
        }
        if int(info["class_count"] or 0) < 1:
            raise ValueError("请先添加至少一个类别")
        if int(info["labeled_count"] or 0) < 1:
            raise ValueError("尚无已标注图片，请先到标注页完成部分标注")
        # 训练门槛放宽：至少 1 张；预标注仍按 20 张在 prelabel 步检查
        return (
            f"标注检查通过：已标注 {info['labeled_count']} · 未标注 {info['unlabeled_count']} · "
            f"类别 {info['class_count']}"
        )

    if step.type == "prelabel":
        root = Path(ds.path)
        info = dataset_prelabel.analyze_prelabel(root, plan.task_type)
        reason = dataset_prelabel.prelabel_block_reason(info)
        if reason:
            raise ValueError(reason)
        job = Job(task_id=task.id, type="prelabel", status="pending", progress=0, message="编排：排队预标注")
        db.add(job)
        db.commit()
        db.refresh(job)
        await submit_job(
            "prelabel",
            {
                "job_id": job.id,
                "task_id": task.id,
                "dataset_id": ds.id,
                "device": "cpu",
                "owner": username_by_id(db, user.id),
            },
        )
        step.job_id = job.id
        step.result = {"job_id": job.id}
        return f"已启动预标注 Job #{job.id}，完成后请抽检"

    if step.type == "review_labels":
        step.result = {"confirmed_by": user.username, "at": _now_iso()}
        return "已记录人工抽检确认，可继续训练"

    if step.type == "apply_config":
        raw = dict(step.params or {})
        raw["pretrained_weight"] = normalize_weight_name(str(raw.get("pretrained_weight") or ""))
        # 规范化设备：探测本机 CUDA，无效/无卡时回退 CPU
        raw["device"] = _normalize_train_device(str(raw.get("device") or "cpu"))
        step.params["device"] = raw["device"]
        cfg = TrainConfig.model_validate(raw)
        # 校验权重存在（非演示时更严格，演示也尽量选到）
        try:
            if cfg.pretrained_weight:
                resolve_pretrained_weight(cfg.pretrained_weight, plan.task_type)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        task.config_json = cfg.model_dump_json()
        task.status = "configured"
        task.step = max(task.step or 0, 3)
        db.commit()
        step.result = {"config": cfg.model_dump()}
        return "训练配置已写入"

    if step.type == "split":
        from app.services import yolo_split

        root = Path(ds.path)
        cfg = TrainConfig.model_validate(json.loads(task.config_json or "{}"))
        info = yolo_split.prepare_yolo_split(
            root,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
        task.status = "configured"
        task.step = max(task.step or 0, 4)
        db.commit()
        step.result = info if isinstance(info, dict) else {"ok": True}
        return "数据集划分完成"

    if step.type == "train":
        job = Job(task_id=task.id, type="train", status="pending", progress=0, message="编排：排队训练")
        db.add(job)
        db.commit()
        db.refresh(job)
        await submit_job("train", {"job_id": job.id, "task_id": task.id})
        step.job_id = job.id
        step.result = {"job_id": job.id}
        return f"已启动训练 Job #{job.id}"

    if step.type == "eval":
        if task.status not in {"trained", "evaluated", "exported"} and not task.model_path:
            raise ValueError("请先完成训练再评估")
        job = Job(task_id=task.id, type="eval", status="pending", progress=0, message="编排：排队评估")
        db.add(job)
        db.commit()
        db.refresh(job)
        await submit_job("eval", {"job_id": job.id, "task_id": task.id})
        step.job_id = job.id
        step.result = {"job_id": job.id}
        return f"已启动评估 Job #{job.id}"

    if step.type == "export":
        if task.status not in {"trained", "evaluated", "exported"} and not task.model_path:
            raise ValueError("请先完成训练再导出")
        formats = step.params.get("formats") or ["pt", "onnx"]
        job = Job(task_id=task.id, type="export", status="pending", progress=0, message="编排：排队导出")
        db.add(job)
        db.commit()
        db.refresh(job)
        await submit_job(
            "export",
            {"job_id": job.id, "task_id": task.id, "formats": formats},
        )
        step.job_id = job.id
        step.result = {"job_id": job.id, "formats": formats}
        return f"已启动导出 Job #{job.id}"

    if step.type == "summary":
        # 兼容旧计划中的摘要步；新计划不再生成该步骤
        step.result = {
            "task_id": plan.task_id,
            "dataset_id": plan.dataset_id,
            "task_type": plan.task_type,
        }
        return f"编排完成。训练任务 #{plan.task_id}，可在模型库查看产物。"

    raise ValueError(f"未知步骤类型：{step.type}")


def ack_job_step(plan: AgentPlanOut, job: Job) -> AgentPlanOut:
    """Job 结束后确认当前异步步骤。"""
    step = plan.steps[plan.current_index]
    if step.job_id != job.id:
        raise ValueError("Job 与当前步骤不匹配")
    if job.status == "completed":
        step.status = "completed"
        step.result = {**(step.result or {}), "job_status": job.status, "job_message": job.message}
        _advance(plan)
    elif job.status in {"failed", "cancelled"}:
        step.status = "failed"
        step.error = job.message or job.status
        plan.status = "failed"
    else:
        raise ValueError(f"Job 尚未结束：{job.status}")
    save_plan(plan)
    return plan


def cancel_plan(plan: AgentPlanOut) -> AgentPlanOut:
    plan.status = "cancelled"
    save_plan(plan)
    return plan


# 允许用户从计划中移除的可选步骤（核心链路不可删）
_REMOVABLE_STEP_TYPES = frozenset({"prelabel", "review_labels", "eval", "export", "summary"})


def patch_plan(plan: AgentPlanOut, patch: dict[str, Any]) -> AgentPlanOut:
    """
    微调计划：修改尚未写入的训练配置、移除未执行的可选步骤。
    已完成/进行中的步骤不可改删。
    """
    if plan.status in {"completed", "cancelled"}:
        raise ValueError("计划已结束，无法修改")

    # 改 apply_config（仅当该步尚未完成）
    cfg_step = next((s for s in plan.steps if s.type == "apply_config"), None)
    if cfg_step and cfg_step.status not in {"completed", "running", "skipped"}:
        changed = False
        if patch.get("epochs") is not None:
            cfg_step.params["epochs"] = int(patch["epochs"])
            changed = True
        if patch.get("batch") is not None:
            cfg_step.params["batch"] = int(patch["batch"])
            changed = True
        if patch.get("device"):
            cfg_step.params["device"] = _normalize_train_device(str(patch["device"]))
            changed = True
        if patch.get("pretrained_weight"):
            name = normalize_weight_name(str(patch["pretrained_weight"]))
            try:
                resolve_pretrained_weight(name, plan.task_type)
            except FileNotFoundError as e:
                raise ValueError(str(e)) from e
            cfg_step.params["pretrained_weight"] = name
            changed = True
        if changed:
            epochs = cfg_step.params.get("epochs")
            batch = cfg_step.params.get("batch")
            device = cfg_step.params.get("device")
            weight = cfg_step.params.get("pretrained_weight")
            cfg_step.description = (
                f"epochs={epochs}，batch={batch}，device={device}，权重={weight}，策略=stable。"
            )

    remove_ids = [str(x) for x in (patch.get("remove_step_ids") or []) if str(x).strip()]
    if remove_ids:
        remove_set = set(remove_ids)
        kept: list[AgentStepOut] = []
        removed_titles: list[str] = []
        for s in plan.steps:
            if s.id not in remove_set:
                kept.append(s)
                continue
            if s.type not in _REMOVABLE_STEP_TYPES:
                raise ValueError(f"步骤「{s.title}」属于核心链路，不能删除")
            if s.status in {"completed", "running"}:
                raise ValueError(f"步骤「{s.title}」已在执行或已完成，不能删除")
            removed_titles.append(s.title)
        if len(kept) < 2:
            raise ValueError("删除后步骤过少，请至少保留核心训练链路")
        # 调整 current_index：指向仍存在的当前步；若当前被删则落到下一未完成步
        cur_id = None
        if 0 <= plan.current_index < len(plan.steps):
            cur_id = plan.steps[plan.current_index].id
        plan.steps = kept
        if cur_id and any(s.id == cur_id for s in plan.steps):
            plan.current_index = next(i for i, s in enumerate(plan.steps) if s.id == cur_id)
        else:
            # 找第一个未完成步骤
            idx = next(
                (
                    i
                    for i, s in enumerate(plan.steps)
                    if s.status not in {"completed", "skipped"}
                ),
                len(plan.steps) - 1,
            )
            plan.current_index = idx
        _mark_ready(plan)
        if removed_titles:
            hint = "已移除步骤：" + "、".join(removed_titles)
            plan.hints = [*(plan.hints or []), hint]

    # 刷新摘要里的 epochs（若仍有 apply_config）
    cfg_step = next((s for s in plan.steps if s.type == "apply_config"), None)
    if cfg_step:
        kind = TASK_TYPE_LABELS.get(plan.task_type, plan.task_type)
        has_eval = any(s.type == "eval" for s in plan.steps)
        has_export = any(s.type == "export" for s in plan.steps)
        has_pre = any(s.type == "prelabel" for s in plan.steps)
        plan.summary = (
            f"将按确认逐步执行：{kind} · {cfg_step.params.get('epochs')} epochs · "
            f"评估={'开' if has_eval else '关'} · 导出={'开' if has_export else '关'}"
            + (" · 预标注开" if has_pre else "")
            + f" · 权重 {cfg_step.params.get('pretrained_weight')}"
        )

    if plan.status == "failed":
        # 微调后允许继续
        cur = plan.steps[plan.current_index] if plan.steps else None
        if cur and cur.status in {"ready", "pending", "failed"}:
            plan.status = "draft" if plan.current_index == 0 and cur.status != "failed" else "running"
            if cur.status == "failed":
                plan.status = "failed"

    save_plan(plan)
    return plan
