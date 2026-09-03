"""对话式 Agent 可用工具：定义（OpenAI tools）与本地执行。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from app.core.task_types import TASK_TYPE_LABELS, normalize_task_type
from app.models.dataset import Dataset
from app.models.user import User
from app.services import agent_platform_ops, agent_train_ops, dataset_prelabel, task_storage

# 工具名 -> 中文说明（前端展示用）
TOOL_LABELS: dict[str, str] = {
    "list_datasets": "列出数据集",
    "inspect_dataset": "检查数据集标注",
    "list_weights": "列出预训练权重",
    "create_dataset": "创建数据集",
    "guide_dataset_upload": "引导上传数据",
    "open_annotate": "打开标注",
    "start_prelabel": "启动预标注",
    "ensure_train_task": "准备训练任务",
    "set_train_config": "设置训练参数",
    "start_train": "开始训练",
    "get_job_status": "查询任务进度",
    "start_eval": "开始评估",
    "run_sample_infer": "抽样推理验证",
    "list_models": "列出模型库",
    "export_model_onnx": "导出 ONNX",
    "cancel_job": "取消任务",
    "list_devices": "探测训练设备",
    "ask_user": "向用户确认",
}


def openai_tool_schemas() -> list[dict[str, Any]]:
    """供 chat/completions 的 tools 参数。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_datasets",
                "description": "列出当前用户可见的数据集。可按任务类型过滤：detect/segment/pose。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "description": "任务类型：detect | segment | pose；省略则分别列出三类摘要",
                            "enum": ["detect", "segment", "pose"],
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_dataset",
                "description": "检查某个数据集的标注门槛：图片数、已标注/未标注、类别数，并判断是否可训练。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "integer", "description": "数据集 ID"},
                        "dataset_name": {
                            "type": "string",
                            "description": "数据集名称（与 dataset_id 二选一，优先 id）",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "按名称查找时的任务类型，默认 detect",
                            "enum": ["detect", "segment", "pose"],
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_weights",
                "description": "列出指定任务类型下可用的预训练权重文件名。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "description": "detect | segment | pose",
                            "enum": ["detect", "segment", "pose"],
                        }
                    },
                    "required": ["task_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_dataset",
                "description": (
                    "创建新的空数据集（detect/segment/pose）。"
                    "调用前应先与用户确认名称与任务类型；系统也会强制二次确认，未确认不会真正创建。"
                    "创建后引导用户在对话内上传图片。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "数据集名称"},
                        "task_type": {
                            "type": "string",
                            "enum": ["detect", "segment", "pose"],
                            "description": "默认 detect",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "guide_dataset_upload",
                "description": "为已有数据集生成上传图片/导入的跳转引导。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "integer"},
                        "dataset_name": {"type": "string"},
                        "task_type": {"type": "string", "enum": ["detect", "segment", "pose"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_annotate",
                "description": "引导用户在对话页右侧打开标注面板（返回按钮）。标注不足不能训练时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "integer"},
                        "dataset_name": {"type": "string"},
                        "task_type": {"type": "string", "enum": ["detect", "segment", "pose"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_prelabel",
                "description": "对数据集启动 AI 预标注 Job。需满足预标注门槛；完成后建议用户抽检。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "integer"},
                        "dataset_name": {"type": "string"},
                        "task_type": {"type": "string", "enum": ["detect", "segment", "pose"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ensure_train_task",
                "description": "为指定数据集准备/复用一个训练任务，返回 task_id。开训前必须先调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "integer"},
                        "dataset_name": {"type": "string"},
                        "task_type": {
                            "type": "string",
                            "enum": ["detect", "segment", "pose"],
                        },
                        "task_id": {
                            "type": "integer",
                            "description": "若已知任务 ID，可直接复用",
                        },
                        "task_name": {"type": "string", "description": "新建任务时的名称（可选）"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_train_config",
                "description": "写入训练超参：epochs/batch/device/pretrained_weight。需已有 task_id。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "epochs": {"type": "integer"},
                        "batch": {"type": "integer"},
                        "device": {
                            "type": "string",
                            "description": "cpu 或 cuda:0",
                        },
                        "pretrained_weight": {
                            "type": "string",
                            "description": "权重仓库中的文件名，如 yolov8n.pt",
                        },
                        "imgsz": {"type": "integer"},
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_train",
                "description": (
                    "尝试启动训练。系统会强制要求用户确认方案；未确认时不会真正开训，"
                    "而是返回确认卡片（确认开训 / 我想改参数）。"
                    "用户确认后由系统自动开训。返回 wait_for_job=true 时停止继续调工具。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_job_status",
                "description": "查询训练/评估 Job 的进度与状态。",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "integer"}},
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_eval",
                "description": (
                    "尝试启动评估。系统会先要求用户确认；未确认不会真正评估。"
                    "用户确认后由系统自动启动。返回 wait_for_job=true 时需等待完成。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_sample_infer",
                "description": "用训练产物或模型库权重，对数据集中一张样例图做推理，判断模型是否基本可用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "model_id": {"type": "integer", "description": "模型库 ID，可选"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_models",
                "description": "列出模型库中已归档的训练模型。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["detect", "segment", "pose"],
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_model_onnx",
                "description": "将模型库中的 PT 转为 ONNX（异步 Job）。可传 model_id 或 task_id。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "integer"},
                        "task_id": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_job",
                "description": "取消进行中的训练/评估/导出/预标注 Job。",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "integer"}},
                    "required": ["job_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_devices",
                "description": "探测本机可用训练设备（CPU / CUDA）。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": (
                    "当信息不完整、标注不足、或需要用户做选择/确认时调用。"
                    "调用后应停止继续调工具，等待用户在对话中回复。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "向用户提出的问题（中文）"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "可选的快捷选项",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]


def _find_dataset(
    db: Session,
    user: User,
    *,
    dataset_id: int | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> Dataset:
    return agent_train_ops.get_dataset(
        db,
        user,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        task_type=task_type,
    )


def _list_datasets_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    raw_tt = args.get("task_type")
    types = [normalize_task_type(str(raw_tt))] if raw_tt else list(TASK_TYPE_LABELS.keys())
    groups: list[dict[str, Any]] = []
    for tt in types:
        q = db.query(Dataset).filter(Dataset.task_type == tt)
        if user.role != "admin":
            q = q.filter(Dataset.owner_id == user.id)
        rows = q.order_by(Dataset.id.desc()).limit(50).all()
        groups.append(
            {
                "task_type": tt,
                "task_type_label": TASK_TYPE_LABELS.get(tt, tt),
                "count": len(rows),
                "items": [
                    {"id": r.id, "name": r.name, "image_count": int(r.image_count or 0)}
                    for r in rows
                ],
            }
        )
    return {"datasets": groups}


def _inspect_dataset_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    ds = _find_dataset(
        db,
        user,
        dataset_id=int(args["dataset_id"]) if args.get("dataset_id") is not None else None,
        dataset_name=str(args.get("dataset_name") or "") or None,
        task_type=str(args.get("task_type") or "") or None,
    )
    root = Path(ds.path)
    info = dataset_prelabel.analyze_prelabel(root, ds.task_type or "detect")
    labeled = int(info.get("labeled_count") or 0)
    unlabeled = int(info.get("unlabeled_count") or 0)
    class_count = int(info.get("class_count") or 0)
    can_train = labeled >= 1 and class_count >= 1
    advice = "可以开始训练配置" if can_train else "标注不足或缺少类别，请先到对应向导完成标注后再继续"
    tt = ds.task_type or "detect"
    actions = []
    if not can_train:
        actions = [
            {
                "type": "upload",
                "label": "在对话中上传图片",
                "dataset_id": ds.id,
            },
            {
                "type": "annotate",
                "label": "去标注",
                "dataset_id": int(ds.id),
                "task_type": tt,
            },
        ]
    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "task_type": ds.task_type,
        "path": ds.path,
        "labeled_count": labeled,
        "unlabeled_count": unlabeled,
        "class_count": class_count,
        "min_labeled": info.get("min_labeled"),
        "can_train": can_train,
        "advice": advice,
        "actions": actions,
    }


def _list_weights_tool(_db: Session, _user: User, args: dict[str, Any]) -> dict[str, Any]:
    tt = normalize_task_type(str(args.get("task_type") or "detect"))
    files = task_storage.list_weight_files(tt)
    return {
        "task_type": tt,
        "count": len(files),
        "weights": [{"name": x["name"], "size": x.get("size")} for x in files],
    }


def _ask_user_tool(_db: Session, _user: User, args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question") or "").strip()
    if not question:
        raise ValueError("ask_user 需要 question")
    options = args.get("options") if isinstance(args.get("options"), list) else []
    return {
        "asked": True,
        "question": question,
        "options": [str(x) for x in options if str(x).strip()],
        "wait_for_user": True,
    }


def _ensure_train_task_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return agent_train_ops.ensure_train_task(
        db,
        user,
        dataset_id=int(args["dataset_id"]) if args.get("dataset_id") is not None else None,
        dataset_name=str(args.get("dataset_name") or "") or None,
        task_type=str(args.get("task_type") or "") or None,
        task_name=str(args.get("task_name") or "") or None,
        task_id=int(args["task_id"]) if args.get("task_id") is not None else None,
    )


def _set_train_config_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    if args.get("task_id") is None:
        raise ValueError("set_train_config 需要 task_id")
    return agent_train_ops.set_train_config(
        db,
        user,
        task_id=int(args["task_id"]),
        epochs=int(args["epochs"]) if args.get("epochs") is not None else None,
        batch=int(args["batch"]) if args.get("batch") is not None else None,
        device=str(args["device"]) if args.get("device") is not None else None,
        pretrained_weight=str(args["pretrained_weight"]) if args.get("pretrained_weight") is not None else None,
        imgsz=int(args["imgsz"]) if args.get("imgsz") is not None else None,
    )


async def _start_train_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    if args.get("task_id") is None:
        raise ValueError("start_train 需要 task_id")
    return await agent_train_ops.start_train(db, user, task_id=int(args["task_id"]))


def _get_job_status_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    if args.get("job_id") is None:
        raise ValueError("get_job_status 需要 job_id")
    return agent_train_ops.get_job_status(db, user, job_id=int(args["job_id"]))


async def _start_eval_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    if args.get("task_id") is None:
        raise ValueError("start_eval 需要 task_id")
    return await agent_train_ops.start_eval(db, user, task_id=int(args["task_id"]))


def _run_sample_infer_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return agent_train_ops.run_sample_infer(
        db,
        user,
        task_id=int(args["task_id"]) if args.get("task_id") is not None else None,
        model_id=int(args["model_id"]) if args.get("model_id") is not None else None,
    )


def _create_dataset_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("create_dataset 需要 name")
    return agent_platform_ops.create_dataset(
        db,
        user,
        name=name,
        task_type=str(args.get("task_type") or "detect"),
    )


def _guide_upload_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return agent_platform_ops.guide_dataset_upload(
        db,
        user,
        dataset_id=int(args["dataset_id"]) if args.get("dataset_id") is not None else None,
        dataset_name=str(args.get("dataset_name") or "") or None,
        task_type=str(args.get("task_type") or "") or None,
    )


def _open_annotate_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return agent_platform_ops.open_annotate(
        db,
        user,
        dataset_id=int(args["dataset_id"]) if args.get("dataset_id") is not None else None,
        dataset_name=str(args.get("dataset_name") or "") or None,
        task_type=str(args.get("task_type") or "") or None,
    )


async def _start_prelabel_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return await agent_platform_ops.start_prelabel(
        db,
        user,
        dataset_id=int(args["dataset_id"]) if args.get("dataset_id") is not None else None,
        dataset_name=str(args.get("dataset_name") or "") or None,
        task_type=str(args.get("task_type") or "") or None,
    )


def _list_models_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return agent_platform_ops.list_models(
        db,
        user,
        task_type=str(args.get("task_type") or "") or None,
    )


async def _export_onnx_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    return await agent_platform_ops.export_model_onnx(
        db,
        user,
        model_id=int(args["model_id"]) if args.get("model_id") is not None else None,
        task_id=int(args["task_id"]) if args.get("task_id") is not None else None,
    )


def _cancel_job_tool(db: Session, user: User, args: dict[str, Any]) -> dict[str, Any]:
    if args.get("job_id") is None:
        raise ValueError("cancel_job 需要 job_id")
    return agent_platform_ops.cancel_job(db, user, job_id=int(args["job_id"]))


def _list_devices_tool(_db: Session, _user: User, _args: dict[str, Any]) -> dict[str, Any]:
    return agent_platform_ops.list_devices()


ToolHandler = Callable[[Session, User, dict[str, Any]], Any]

_HANDLERS: dict[str, ToolHandler] = {
    "list_datasets": _list_datasets_tool,
    "inspect_dataset": _inspect_dataset_tool,
    "list_weights": _list_weights_tool,
    "create_dataset": _create_dataset_tool,
    "guide_dataset_upload": _guide_upload_tool,
    "open_annotate": _open_annotate_tool,
    "start_prelabel": _start_prelabel_tool,
    "ensure_train_task": _ensure_train_task_tool,
    "set_train_config": _set_train_config_tool,
    "start_train": _start_train_tool,
    "get_job_status": _get_job_status_tool,
    "start_eval": _start_eval_tool,
    "run_sample_infer": _run_sample_infer_tool,
    "list_models": _list_models_tool,
    "export_model_onnx": _export_onnx_tool,
    "cancel_job": _cancel_job_tool,
    "list_devices": _list_devices_tool,
    "ask_user": _ask_user_tool,
}


def _strip_heavy_fields(data: dict[str, Any]) -> dict[str, Any]:
    """去掉不宜塞进 LLM 上下文的大字段。"""
    out = dict(data)
    out.pop("preview_base64", None)
    out.pop("image_base64", None)
    return out


async def execute_tool(
    name: str,
    arguments_json: str,
    *,
    db: Session,
    user: User,
) -> dict[str, Any]:
    """执行工具并返回可 JSON 序列化的结果。"""
    handler = _HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"未知工具：{name}"}
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        args = {}
    try:
        raw = handler(db, user, args)
        data = await raw if inspect.isawaitable(raw) else raw
        if not isinstance(data, dict):
            data = {"value": data}
        return {"ok": True, "tool": name, "data": data}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "tool": name, "error": str(e)}


def tool_content_for_llm(exec_result: dict[str, Any]) -> str:
    """写入 messages 的 tool content（去掉预览图等大字段）。"""
    payload = dict(exec_result)
    data = payload.get("data")
    if isinstance(data, dict):
        payload["data"] = _strip_heavy_fields(data)
    return json.dumps(payload, ensure_ascii=False)
