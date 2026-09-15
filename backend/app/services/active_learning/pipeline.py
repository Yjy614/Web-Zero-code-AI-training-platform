"""主动学习流水线：建会话、筛图、并入原集、准备续训。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.train import ModelRecord, TrainTask
from app.models.user import User
from app.services import dataset_annotate, dataset_storage, model_artifacts
from app.services.active_learning import session_store
from app.services.active_learning.strategy import get_strategy
from app.services.active_learning.types import DetectionBox, ScreenSummary
from app.services.bootstrap import username_by_id


ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]


def _get_owned_dataset(db: Session, user: User, dataset_id: int) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
    if not ds:
        raise FileNotFoundError("数据集不存在")
    if user.role != "admin" and ds.owner_id != user.id:
        raise PermissionError("无权访问该数据集")
    return ds


def _get_accessible_model(db: Session, user: User, model_id: int) -> ModelRecord:
    row = db.query(ModelRecord).filter(ModelRecord.id == int(model_id)).first()
    if not row:
        raise FileNotFoundError("模型不存在")
    if user.role != "admin" and row.owner_id != user.id:
        raise PermissionError("无权访问该模型")
    return row


def default_target_dataset_id(db: Session, model: ModelRecord) -> int | None:
    """从模型关联训练任务或 metrics.lineage 推断原数据集。"""
    if model.task_id:
        task = db.query(TrainTask).filter(TrainTask.id == model.task_id).first()
        if task and task.dataset_id:
            return int(task.dataset_id)
    # 兼容：直接读登记时写入的 lineage
    try:
        metrics = json.loads(model.metrics_json or "{}")
    except json.JSONDecodeError:
        metrics = {}
    if isinstance(metrics, dict):
        lin = metrics.get("lineage")
        if isinstance(lin, dict) and lin.get("dataset_id") is not None:
            try:
                return int(lin["dataset_id"])
            except (TypeError, ValueError):
                return None
    return None


def create_al_session(
    db: Session,
    user: User,
    *,
    model_id: int,
    target_dataset_id: int | None = None,
) -> dict[str, Any]:
    """创建主动学习会话，并建 staging 数据集（复制类别）。原集强制取自模型训练关联。"""
    from app.services.active_learning.staging import abandon_other_active_sessions, cleanup_orphaned_staging

    model = _get_accessible_model(db, user, model_id)
    tt = (model.task_type or "detect").strip().lower()
    from app.services.active_learning.strategy import SUPPORTED_TASK_TYPES

    if tt not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"主动学习不支持任务类型：{tt}")

    pt = model_artifacts.find_pt_path(model)
    if pt is None:
        raise FileNotFoundError("该模型没有可用的 PT 权重")

    # 新建前：取消同模型进行中会话 + 清理悬空临时集
    abandon_other_active_sessions(db, user, model_id=int(model.id))
    cleanup_orphaned_staging(db, user)

    # 强制原集：以模型关联为准；若调用方传入则必须一致
    tid = default_target_dataset_id(db, model)
    if tid is None:
        raise ValueError("找不到该模型的原训练数据集，无法开始主动学习")
    if target_dataset_id is not None and int(target_dataset_id) != int(tid):
        raise ValueError("主动学习必须使用该模型的原训练数据集，不能更换为其它数据集")
    target = _get_owned_dataset(db, user, int(tid))
    if (target.task_type or "detect").strip().lower() != tt:
        raise ValueError("原数据集任务类型与模型不一致")

    try:
        classes = json.loads(target.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    if not isinstance(classes, list) or not classes:
        raise ValueError("原数据集尚未配置类别，请先完成标注类别设置")

    # 创建 staging 数据集
    username = user.username
    staging_name = dataset_storage.validate_dataset_name(f"AL_{target.name}_{session_store.new_session_id()}")
    root = dataset_storage.ensure_dataset_dirs(username, staging_name, owner_id=user.id, task_type=tt)
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes
    meta["active_learn"] = True
    meta["active_learn_target_id"] = target.id
    # 姿态：复制原集骨架配置，保证复核与续训关键点数一致
    if tt == "pose" and target.path:
        try:
            tmeta = dataset_storage.read_meta(Path(target.path))
            if isinstance(tmeta.get("pose"), dict):
                meta["pose"] = tmeta["pose"]
        except Exception:  # noqa: BLE001
            pass
    dataset_storage.write_meta(root, meta)

    staging = Dataset(
        name=staging_name,
        path=str(root),
        task_type=tt,
        owner_id=user.id,
        classes_json=json.dumps(classes, ensure_ascii=False),
        image_count=0,
    )
    db.add(staging)
    db.commit()
    db.refresh(staging)

    session = session_store.create_session(
        user.id,
        {
            "model_id": model.id,
            "model_name": model.name,
            "task_type": tt,
            "target_dataset_id": target.id,
            "target_dataset_name": target.name,
            "staging_dataset_id": staging.id,
            "staging_dataset_name": staging.name,
        },
    )
    return session


def run_screen_job(
    db: Session,
    *,
    user_id: int,
    session_id: str,
    update_progress: ProgressCb | None = None,
    is_cancelled: CancelCb | None = None,
) -> dict[str, Any]:
    """对 staging 数据集全量推理并划分难易例；easy 写入草稿标注。"""

    def _prog(p: float, msg: str) -> None:
        if update_progress:
            update_progress(p, msg)

    def _cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    session = session_store.load_session(user_id, session_id)
    session["status"] = "screening"
    session["error"] = None
    session_store.save_session(user_id, session)

    model = db.query(ModelRecord).filter(ModelRecord.id == int(session["model_id"])).first()
    if not model:
        raise FileNotFoundError("模型不存在")
    staging = db.query(Dataset).filter(Dataset.id == int(session["staging_dataset_id"])).first()
    if not staging or not staging.path:
        raise FileNotFoundError("临时数据集不存在")

    pt = model_artifacts.find_pt_path(model)
    if pt is None:
        raise FileNotFoundError("找不到模型 PT 权重")

    root = Path(staging.path)
    images = dataset_storage.list_images(root, include_removed=False)
    if not images:
        raise ValueError("请先上传本轮新图片")

    strategy = get_strategy(session.get("task_type") or "detect")
    tt = strategy.task_type

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("未安装 ultralytics，无法筛图") from e

    _prog(5, "正在加载模型…")
    yolo = YOLO(str(pt))
    items: list[dict[str, Any]] = []
    written = 0
    easy_n = hard_n = empty_n = 0

    total = len(images)
    for idx, name in enumerate(images):
        if _cancelled():
            raise InterruptedError("筛图已取消")
        img_path = root / "images" / name
        if not img_path.is_file():
            continue
        # 低阈值尽量检出，再由策略划分
        results = yolo.predict(source=str(img_path), conf=0.15, iou=0.45, imgsz=640, verbose=False)
        r0 = results[0] if results else None
        dets: list[DetectionBox] = []
        width = height = 0
        if r0 is not None:
            try:
                from PIL import Image

                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001
                if hasattr(r0, "orig_shape") and r0.orig_shape is not None:
                    height = int(r0.orig_shape[0])
                    width = int(r0.orig_shape[1])

            names = getattr(yolo, "names", None) or getattr(r0, "names", None) or {}
            if not isinstance(names, dict):
                try:
                    names = dict(enumerate(names))
                except Exception:  # noqa: BLE001
                    names = {}
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().tolist()
                confs = r0.boxes.conf.cpu().tolist()
                clss = r0.boxes.cls.cpu().tolist()
                polygons: list[list[list[float]] | None] = [None] * len(xyxy)
                if getattr(r0, "masks", None) is not None and getattr(r0.masks, "xy", None) is not None:
                    for i, poly in enumerate(r0.masks.xy):
                        if i >= len(polygons):
                            break
                        try:
                            pts = [[float(p[0]), float(p[1])] for p in poly]
                            polygons[i] = pts if len(pts) >= 3 else None
                        except Exception:  # noqa: BLE001
                            polygons[i] = None
                kpts_list: list[list[list[float]] | None] = [None] * len(xyxy)
                if getattr(r0, "keypoints", None) is not None:
                    try:
                        kxy = r0.keypoints.xy.cpu().tolist()
                        kconf = None
                        if getattr(r0.keypoints, "conf", None) is not None:
                            kconf = r0.keypoints.conf.cpu().tolist()
                        for i, pts in enumerate(kxy):
                            if i >= len(kpts_list):
                                break
                            row: list[list[float]] = []
                            for j, p in enumerate(pts):
                                x, y = float(p[0]), float(p[1])
                                v = 2.0
                                if kconf is not None and i < len(kconf) and j < len(kconf[i]):
                                    c = float(kconf[i][j])
                                    v = 2.0 if c >= 0.5 else (1.0 if c > 0.01 else 0.0)
                                row.append([x, y, v])
                            kpts_list[i] = row
                    except Exception:  # noqa: BLE001
                        pass

                for i, box in enumerate(xyxy):
                    cid = int(clss[i])
                    cname = str(names.get(cid, names.get(str(cid), f"class_{cid}")))
                    dets.append(
                        DetectionBox(
                            class_id=cid,
                            class_name=cname,
                            confidence=float(confs[i]),
                            bbox_xyxy=[float(v) for v in box],
                            polygon=polygons[i],
                            keypoints=kpts_list[i],
                        )
                    )

        score = strategy.score_image(image_name=name, detections=dets, image_size=(width, height))
        if score.difficulty == "easy":
            easy_n += 1
            if width > 0 and height > 0:
                payload = strategy.detections_to_label_payload(
                    score, image_width=width, image_height=height
                )
                if payload:
                    if tt == "detect":
                        dataset_annotate.write_annotations(root, name, payload)
                        written += 1
                    elif tt == "segment":
                        dataset_annotate.write_polygons(root, name, payload)
                        written += 1
                    elif tt == "pose":
                        dataset_annotate.write_poses(root, name, payload)
                        written += 1
        elif score.difficulty == "hard":
            hard_n += 1
        else:
            empty_n += 1
            # empty 也当难例优先给人看
            hard_n += 1
            score.difficulty = "hard"
            score.reason = score.reason or "未检出，建议人工确认"

        items.append(score.to_dict())
        _prog(5 + 90 * (idx + 1) / total, f"筛图中 {idx + 1}/{total}")

    # 难例靠前
    items.sort(key=lambda x: (-(1 if x.get("difficulty") == "hard" else 0), -float(x.get("score") or 0)))

    summary = ScreenSummary(
        total=total,
        easy_count=easy_n,
        hard_count=hard_n,
        empty_count=empty_n,
        written_labels=written,
    )
    session["items"] = items
    session["summary"] = summary.to_dict()
    session["status"] = "screened"
    session_store.save_session(user_id, session)
    _prog(100, "筛图完成")
    return {"session": session, "summary": summary.to_dict()}


def merge_staging_into_target(db: Session, user: User, session_id: str) -> dict[str, Any]:
    """将 staging 图片与标注并入原数据集。"""
    session = session_store.load_session(user.id, session_id)
    if session.get("status") == "merged":
        return {"merged": int(session.get("merged_count") or 0), "session": session, "message": "已并入，无需重复操作"}
    staging = _get_owned_dataset(db, user, int(session["staging_dataset_id"]))
    target = _get_owned_dataset(db, user, int(session["target_dataset_id"]))
    src = Path(staging.path)
    dst = Path(target.path)
    images = dataset_storage.list_images(src, include_removed=False)
    merged = 0
    for name in images:
        sp = src / "images" / name
        if not sp.is_file():
            continue
        # 避免重名覆盖：若目标已有同名，加前缀
        dest_name = name
        if (dst / "images" / dest_name).exists():
            dest_name = f"al_{session_id}_{name}"
        dp = dst / "images" / dest_name
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        # 标注
        lp = dataset_storage.label_path_for(src, name)
        if lp.is_file():
            dlp = dataset_storage.label_path_for(dst, dest_name)
            dlp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lp, dlp)
        merged += 1

    # 刷新目标计数
    target.image_count = len(dataset_storage.list_images(dst, include_removed=True))
    db.commit()

    # 并入成功后删除临时集，避免悬空出现在数据集管理
    from app.services.active_learning.staging import delete_staging_dataset, is_active_learn_staging

    if is_active_learn_staging(staging):
        delete_staging_dataset(db, staging)

    session["status"] = "merged"
    session["merged_count"] = merged
    session["staging_dataset_id"] = None
    session["staging_dataset_name"] = None
    session_store.save_session(user.id, session)
    return {"merged": merged, "session": session, "message": f"已并入原数据集 {merged} 张"}


def ensure_finetune_weight_alias(db: Session, model_id: int) -> str:
    """
    把模型库 PT 注册为可训练权重名：@model:{id}
    实际路径在 resolve 时解析。
    """
    row = db.query(ModelRecord).filter(ModelRecord.id == int(model_id)).first()
    if not row:
        raise FileNotFoundError("模型不存在")
    if model_artifacts.find_pt_path(row) is None:
        raise FileNotFoundError("模型无 PT 权重")
    return f"@model:{int(model_id)}"


def prepare_retrain_task(
    db: Session,
    user: User,
    session_id: str,
    *,
    epochs: int = 50,
    batch: int = 8,
    device: str = "cpu",
    imgsz: int = 640,
) -> dict[str, Any]:
    """在原数据集上准备续训任务（权重指向当前模型）。不自动开训。"""
    from app.services import agent_train_ops

    session = session_store.load_session(user.id, session_id)
    if session.get("status") not in {"merged", "screened", "retraining", "done"}:
        raise ValueError("请先完成筛图；建议复核并并入原数据集后再续训")
    if session.get("status") == "screened":
        # 允许未点并入时自动并入一次，避免卡住
        merge_staging_into_target(db, user, session_id)
        session = session_store.load_session(user.id, session_id)

    target_id = int(session["target_dataset_id"])
    model_id = int(session["model_id"])
    weight = ensure_finetune_weight_alias(db, model_id)
    tt = session.get("task_type") or "detect"

    ensured = agent_train_ops.ensure_train_task(
        db,
        user,
        dataset_id=target_id,
        task_name=f"{session.get('target_dataset_name') or 'ds'}_al_{session_id}",
    )
    task_id = int(ensured["task_id"])
    agent_train_ops.set_train_config(
        db,
        user,
        task_id=task_id,
        epochs=int(epochs),
        batch=int(batch),
        device=str(device or "cpu"),
        pretrained_weight=weight,
        imgsz=int(imgsz),
    )
    session["retrain_task_id"] = task_id
    session["finetune_weight"] = weight
    session["status"] = "retraining"
    session_store.save_session(user.id, session)
    return {
        "task_id": task_id,
        "pretrained_weight": weight,
        "dataset_id": target_id,
        "task_type": tt,
        "session": session,
        "message": "续训任务已准备，请确认后启动训练",
    }
