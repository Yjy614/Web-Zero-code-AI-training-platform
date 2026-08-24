"""JobRunner：本机 Ultralytics 真实训练 / 评估 / 导出。"""

from __future__ import annotations

import json
import shutil
import threading
import traceback
from pathlib import Path
from typing import Any

from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.train import Job, ModelRecord, TrainTask
from app.runners.base import JobRunner
from app.runners import job_control
from app.services import task_storage
from app.services import dataset_prelabel, dataset_storage
from app.services.eval_report import write_eval_reports
from app.services.train_strategy import resolve_train_hparams
from app.services.weight_resolve import resolve_pretrained_weight
from app.services.bootstrap import username_by_id

# 禁止 Ultralytics 因缺依赖而自动联网安装 / 更新
import os

os.environ.setdefault("YOLO_OFFLINE", "true")


class LocalJobRunner(JobRunner):
    """本机调用 Ultralytics YOLO，写入真实曲线与权重。"""

    async def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = int(payload["job_id"])
        job_control.clear_cancel(job_id)
        t = threading.Thread(target=_run_job_sync, args=(job_id, job_type, payload), daemon=True)
        t.start()
        return str(job_id)

    async def cancel(self, job_id: str) -> None:
        job_control.request_cancel(int(job_id))


def _run_job_sync(job_id: int, job_type: str, payload: dict[str, Any]) -> None:
    db = SessionLocal()
    job_control.mark_job_active(job_id)
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        task = db.query(TrainTask).filter(TrainTask.id == job.task_id).first() if job else None
        if not job or not task:
            return
        job.status = "running"
        job.message = "任务已开始"
        job.progress = 0
        db.commit()

        if job_type == "train":
            _local_train(db, job, task, payload)
        elif job_type == "eval":
            _local_eval(db, job, task, payload)
        elif job_type == "export":
            _local_export(db, job, task, payload)
        elif job_type == "prelabel":
            _local_prelabel(db, job, task, payload)
        else:
            job.status = "failed"
            job.message = f"未知任务类型：{job_type}"
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.message = f"执行失败：{exc}"
            job.result_json = json.dumps(
                {"error": str(exc), "trace": traceback.format_exc()[-2000:]},
                ensure_ascii=False,
            )
            db.commit()
        task = db.query(TrainTask).filter(TrainTask.id == job.task_id).first() if job else None
        if task and task.status == "training":
            task.status = "failed"
            db.commit()
    finally:
        job_control.mark_job_inactive(job_id)
        db.close()


def _parse_config(task: TrainTask) -> dict[str, Any]:
    try:
        cfg = json.loads(task.config_json or "{}")
        return cfg if isinstance(cfg, dict) else {}
    except json.JSONDecodeError:
        return {}


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "未安装 ultralytics。请在 backend 环境执行：pip install ultralytics"
        ) from e
    from ultralytics import YOLO

    return YOLO


def _dataset_yaml(db, task: TrainTask) -> Path:
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds or not ds.path:
        raise FileNotFoundError("关联数据集不存在")
    root = Path(ds.path)
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError("缺少 data.yaml，请先在配置步完成数据划分")
    return yaml_path


def _upsert_model(
    db, task: TrainTask, weights_best: Path, metrics: dict[str, Any], task_type: str = "detect"
) -> None:
    """兼容旧名：改为始终登记新模型，不覆盖同任务历史记录。"""
    _register_model(db, task, weights_best, metrics, task_type)


def _register_model(
    db,
    task: TrainTask,
    weights_best: Path,
    metrics: dict[str, Any],
    task_type: str = "detect",
    *,
    model_name: str | None = None,
) -> str:
    """
    每次训练新增一条模型库记录；权重复制到独立归档路径：
    models/<tt>/<user>/<task_name>/<run_key>/<run_key>.pt
    """
    from app.services import model_artifacts

    username = username_by_id(db, task.owner_id)
    stems = task_storage.collect_archived_model_stems(username, task.name, task_type)
    name = model_name or model_artifacts.next_unique_model_name(
        db, task.owner_id, task.name, archived_stems=stems
    )
    dest_dir = task_storage.versioned_model_dir(username, task.name, name, task_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.pt"
    # 若目标已存在则再顺延，避免覆盖
    if dest.exists():
        stems.add(name)
        name = model_artifacts.next_unique_model_name(
            db, task.owner_id, task.name, archived_stems=stems
        )
        dest_dir = task_storage.versioned_model_dir(username, task.name, name, task_type)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.pt"
    shutil.copy2(weights_best, dest)
    payload = json.dumps(metrics, ensure_ascii=False)
    db.add(
        ModelRecord(
            name=name,
            path=str(dest),
            task_type=task_type,
            owner_id=task.owner_id,
            task_id=task.id,
            metrics_json=payload,
        )
    )
    return name


def _local_train(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    YOLO = _require_ultralytics()
    cfg = _parse_config(task)
    epochs = int(payload.get("epochs") or cfg.get("epochs") or 50)
    epochs = max(1, min(epochs, 500))
    batch = int(cfg.get("batch") or 8)
    imgsz = int(cfg.get("imgsz") or 640)
    device = str(cfg.get("device") or "cpu")
    augment = bool(cfg.get("augment", True))
    weight_name = str(cfg.get("pretrained_weight") or "")
    strategy_hp = resolve_train_hparams(cfg.get("train_strategy"))

    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    task_type = (ds.task_type if ds else None) or "detect"
    weight_path = resolve_pretrained_weight(weight_name, task_type)
    data_yaml = _dataset_yaml(db, task)
    username = username_by_id(db, task.owner_id)
    dirs = task_storage.ensure_task_dirs(username, task.name, task_type)
    # 开训前占用唯一模型名，训练输出与归档都用该名，避免互相覆盖
    from app.services import model_artifacts

    archive_dir = task_storage.saved_models_task_dir(username, task.name, task_type)
    model_name = model_artifacts.next_unique_model_name(
        db,
        task.owner_id,
        task.name,
        archive_dir=archive_dir,
        archived_stems=task_storage.collect_archived_model_stems(username, task.name, task_type),
    )
    runs = dirs["runs"] / model_name
    runs.mkdir(parents=True, exist_ok=True)

    task.status = "training"
    db.commit()

    history: list[dict[str, Any]] = []
    model = YOLO(str(weight_path))
    # 在 on_train_epoch_end 缓存本轮 train loss（on_fit_epoch_end 时 tloss 常已被清空）
    epoch_train_loss: dict[str, float] = {"value": 0.0}

    def _read_loss_from_csv(trainer) -> float | None:  # noqa: ANN001
        csv_path = getattr(trainer, "csv", None)
        if not csv_path:
            return None
        try:
            rows = _parse_results_csv(Path(csv_path))
            if rows:
                return float(rows[-1].get("loss") or 0.0)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _extract_train_loss(trainer) -> float:  # noqa: ANN001
        """尽量从 Ultralytics trainer 取出本轮训练 loss（多分量之和）。"""
        # 1) label_loss_items：与官方日志一致
        try:
            tloss = getattr(trainer, "tloss", None)
            label_fn = getattr(trainer, "label_loss_items", None)
            if tloss is not None and callable(label_fn):
                items = label_fn(tloss, prefix="train")
                if isinstance(items, dict) and items:
                    vals = [float(v) for v in items.values() if v is not None]
                    if vals:
                        return float(sum(vals))
        except Exception:  # noqa: BLE001
            pass
        # 2) 直接对 tloss 求和
        try:
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                if hasattr(tloss, "detach"):
                    tloss = tloss.detach().float().cpu()
                if hasattr(tloss, "sum"):
                    return float(tloss.sum().item())
                if hasattr(tloss, "mean"):
                    return float(tloss.mean().item())
                return float(tloss)
        except Exception:  # noqa: BLE001
            pass
        # 3) metrics / csv 兜底
        metrics = getattr(trainer, "metrics", None) or {}
        if isinstance(metrics, dict):
            parts: list[float] = []
            for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "train/loss"):
                if key in metrics and metrics[key] is not None:
                    try:
                        parts.append(float(metrics[key]))
                    except (TypeError, ValueError):
                        pass
            if parts:
                return float(sum(parts))
        csv_loss = _read_loss_from_csv(trainer)
        if csv_loss is not None:
            return csv_loss
        return float(epoch_train_loss["value"] or 0.0)

    def on_train_epoch_end(trainer) -> None:  # noqa: ANN001
        # 训练阶段结束时 tloss 仍有效，先缓存
        epoch_train_loss["value"] = _extract_train_loss(trainer)

    def on_fit_epoch_end(trainer) -> None:  # noqa: ANN001
        if job_control.is_cancelled(job.id):
            trainer.stop = True
            return
        # Ultralytics 的 epoch 为 0-based；全部轮次结束后还会再回调一次（临时 epoch+1 写最终指标）
        raw_ep = int(getattr(trainer, "epoch", 0)) + 1
        is_final_metrics_pass = raw_ep > epochs
        ep = min(raw_ep, epochs)

        loss_v = _extract_train_loss(trainer)
        if loss_v <= 0 and epoch_train_loss["value"] > 0:
            loss_v = float(epoch_train_loss["value"])

        metrics = getattr(trainer, "metrics", None) or {}
        map50 = 0.0
        if isinstance(metrics, dict):
            for key in ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"):
                if key in metrics and metrics[key] is not None:
                    try:
                        map50 = float(metrics[key])
                        break
                    except (TypeError, ValueError):
                        pass

        if is_final_metrics_pass:
            # 收尾回调：不追加虚假 epoch，只刷新末点指标与文案
            if history:
                if map50:
                    history[-1]["map50"] = round(map50, 4)
                if loss_v > 0:
                    history[-1]["loss"] = round(loss_v, 4)
            job.progress = 100.0
            job.message = f"训练收尾中（已完成 {epochs}/{epochs}）…"
        else:
            history.append(
                {
                    "epoch": ep,
                    "loss": round(loss_v, 4),
                    "map50": round(map50, 4),
                }
            )
            job.progress = min(100.0, round(ep / epochs * 100, 1))
            job.message = f"训练中 epoch {ep}/{epochs}"

        job.result_json = json.dumps(
            {
                "history": history,
                "epochs": epochs,
                "demo": False,
                "train_strategy": strategy_hp["train_strategy"],
            },
            ensure_ascii=False,
        )
        db.commit()

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    try:
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(dirs["runs"]),
            name=model_name,
            exist_ok=True,
            augment=augment,
            plots=False,
            save=True,
            verbose=False,
            lr0=strategy_hp["lr0"],
            lrf=strategy_hp["lrf"],
            cos_lr=strategy_hp["cos_lr"],
            warmup_epochs=strategy_hp["warmup_epochs"],
        )
    except Exception:
        if job_control.is_cancelled(job.id):
            job.status = "cancelled"
            job.message = "训练已取消"
            task.status = "configured"
            db.commit()
            return
        raise

    if job_control.is_cancelled(job.id):
        job.status = "cancelled"
        job.message = "训练已取消"
        task.status = "configured"
        db.commit()
        return

    # Ultralytics 输出：runs/<task>/<model_name>/weights/best.pt
    best = runs / "weights" / "best.pt"
    last = runs / "weights" / "last.pt"
    if not best.is_file() and last.is_file():
        shutil.copy2(last, best)
    if not best.is_file():
        raise FileNotFoundError("训练结束但未找到 best.pt，请检查数据集标注与划分是否有效")

    # 若 history 为空（回调未触发），尝试解析 results.csv
    if not history:
        history = _parse_results_csv(runs / "results.csv")

    task.model_path = str(best)
    task.status = "trained"
    task.step = max(task.step, 5)
    best_map = history[-1]["map50"] if history else 0
    best_loss = history[-1]["loss"] if history else 0
    task.metrics_json = json.dumps(
        {
            "best_map50": best_map,
            "history": history,
            "epochs": epochs,
            "demo": False,
            "run_key": model_name,
        },
        ensure_ascii=False,
    )
    _register_model(
        db,
        task,
        best,
        {"map50": best_map, "loss": best_loss, "demo": False, "epochs": epochs},
        task_type,
        model_name=model_name,
    )
    # 训练成功：本轮 AI 预标注视为用户标注（保留框，仅去掉可撤销标记）
    _commit_prelabel_after_train(db, task)

    job.status = "completed"
    job.progress = 100
    job.message = f"本机训练完成（{epochs} epochs）"
    job.result_json = json.dumps(
        {"history": history, "model_path": str(best), "demo": False, "epochs": epochs},
        ensure_ascii=False,
    )
    db.commit()


def _commit_prelabel_after_train(db, task: TrainTask) -> None:
    """训练完成后将预标注确认为用户标注。"""
    if not task.dataset_id:
        return
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds or not ds.path:
        return
    try:
        dataset_prelabel.commit_prelabel_as_user(Path(ds.path))
    except Exception:
        # 不影响训练成功结果
        pass


def _parse_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        import csv

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row.get("epoch") or 0))
                except (TypeError, ValueError):
                    continue
                loss = 0.0
                # 优先：各 train loss 分量之和；否则单列 train/loss / val/box_loss
                parts: list[float] = []
                for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
                    if k in row and row[k] not in (None, ""):
                        try:
                            parts.append(float(row[k]))
                        except ValueError:
                            pass
                if parts:
                    loss = sum(parts)
                else:
                    for k in ("train/loss", "val/box_loss"):
                        if k in row and row[k] not in (None, ""):
                            try:
                                loss = float(row[k])
                                break
                            except ValueError:
                                pass
                map50 = 0.0
                for k in ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"):
                    if k in row and row[k] not in (None, ""):
                        try:
                            map50 = float(row[k])
                            break
                        except ValueError:
                            pass
                rows.append({"epoch": ep + 1 if ep == 0 else ep, "loss": round(loss, 4), "map50": round(map50, 4)})
        return rows
    except OSError:
        return []


def _local_eval(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    _ = payload
    YOLO = _require_ultralytics()
    if not task.model_path or not Path(task.model_path).is_file():
        raise FileNotFoundError("请先完成训练并生成 best.pt")
    data_yaml = _dataset_yaml(db, task)
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    task_type = (ds.task_type if ds else None) or "detect"
    username = username_by_id(db, task.owner_id)
    dirs = task_storage.ensure_task_dirs(username, task.name, task_type)
    run_key = task_storage.resolve_run_key(task_name=task.name, model_path=task.model_path)
    report_dir = task_storage.versioned_reports_dir(username, task.name, run_key, task_type)
    report_dir.mkdir(parents=True, exist_ok=True)
    val_project = dirs["runs"] / run_key
    val_project.mkdir(parents=True, exist_ok=True)

    job.progress = 10
    job.message = "评估中…"
    db.commit()

    if job_control.is_cancelled(job.id):
        job.status = "cancelled"
        job.message = "评估已取消"
        db.commit()
        return

    model = YOLO(task.model_path)
    # 写入本轮 runs/<run_key>/val，避免多次评估互相覆盖、也避免污染 backend/runs
    res = model.val(
        data=str(data_yaml),
        plots=False,
        verbose=False,
        project=str(val_project),
        name="val",
        exist_ok=True,
    )
    box = getattr(res, "box", None)
    map50 = float(getattr(box, "map50", 0) or 0) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0) or 0) if box is not None else 0.0
    precision = float(getattr(box, "mp", 0) or 0) if box is not None else 0.0
    recall = float(getattr(box, "mr", 0) or 0) if box is not None else 0.0

    metrics: dict[str, Any] = {
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "demo": False,
        "run_key": run_key,
        "suggestion": "真实评估完成。可结合 AI 优化建议调整数据量与超参后再次训练。",
    }
    prev: dict[str, Any] = {}
    try:
        prev = json.loads(task.metrics_json or "{}")
        if not isinstance(prev, dict):
            prev = {}
    except json.JSONDecodeError:
        prev = {}
    if prev.get("ai_advice"):
        metrics["ai_advice"] = prev["ai_advice"]
        metrics["ai_advice_source"] = prev.get("ai_advice_source")
    if prev.get("history"):
        metrics["history"] = prev["history"]

    report_json, report_html = write_eval_reports(task.name, metrics, report_dir)
    prev.update(metrics)
    prev["run_key"] = run_key
    task.metrics_json = json.dumps(prev, ensure_ascii=False)
    task.status = "evaluated"
    task.step = max(task.step, 6)

    job.status = "completed"
    job.progress = 100
    job.message = "本机评估完成"
    job.result_json = json.dumps(
        {
            "metrics": metrics,
            "report_json": str(report_json),
            "report_html": str(report_html),
            "run_key": run_key,
            "demo": False,
        },
        ensure_ascii=False,
    )
    db.commit()


def _local_export(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    YOLO = _require_ultralytics()
    # 模型库按卡片导出时可指定 model_path / run_key，避免误用任务上最新训练路径
    src_model = str(payload.get("model_path") or task.model_path or "").strip()
    if not src_model or not Path(src_model).is_file():
        raise FileNotFoundError("请先完成训练并生成 best.pt")
    formats = payload.get("formats") or ["pt", "onnx"]
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    task_type = (ds.task_type if ds else None) or "detect"
    username = username_by_id(db, task.owner_id)
    task_storage.ensure_task_dirs(username, task.name, task_type)
    run_key = str(payload.get("run_key") or "").strip() or task_storage.resolve_run_key(
        task_name=task.name, model_path=src_model
    )
    export_dir = task_storage.versioned_exports_dir(username, task.name, run_key, task_type)
    export_dir.mkdir(parents=True, exist_ok=True)
    side_by_side = bool(payload.get("side_by_side"))
    model = YOLO(src_model)
    files: list[dict[str, Any]] = []
    total = max(len(formats), 1)

    for idx, fmt in enumerate(formats, start=1):
        if job_control.is_cancelled(job.id):
            job.status = "cancelled"
            job.message = "导出已取消"
            db.commit()
            return
        job.message = f"正在导出 {fmt}"
        job.progress = round((idx - 1) / total * 100, 1)
        db.commit()

        fmt_l = str(fmt).lower()
        if fmt_l == "pt":
            out = export_dir / f"{run_key}.pt"
            shutil.copy2(src_model, out)
            rel_name = f"{run_key}/{out.name}"
            files.append(
                {"format": "pt", "path": str(out), "name": rel_name, "demo": False, "run_key": run_key}
            )
        elif fmt_l == "onnx":
            try:
                import onnx  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "缺少 ONNX 导出依赖。请在 web_zero 环境执行："
                    "python -m pip install \"onnx>=1.12.0,<2.0.0\" onnxruntime onnxslim"
                ) from e
            exported = model.export(format="onnx", imgsz=int(_parse_config(task).get("imgsz") or 640))
            src = Path(str(exported))
            out = export_dir / f"{run_key}.onnx"
            if src.is_file():
                shutil.copy2(src, out)
            else:
                cand = Path(src_model).with_suffix(".onnx")
                if cand.is_file():
                    shutil.copy2(cand, out)
                else:
                    raise FileNotFoundError("ONNX 导出失败：未找到输出文件")
            # 模型库旁路：与归档 PT 同目录再放一份，便于 has_onnx / 下载直达
            if side_by_side:
                side = Path(src_model).with_suffix(".onnx")
                try:
                    if side.resolve() != out.resolve():
                        shutil.copy2(out, side)
                except OSError:
                    pass
            rel_name = f"{run_key}/{out.name}"
            files.append(
                {
                    "format": "onnx",
                    "path": str(out),
                    "name": rel_name,
                    "demo": False,
                    "run_key": run_key,
                }
            )
        else:
            raise ValueError(f"不支持的导出格式：{fmt}")

        job.progress = round(idx / total * 100, 1)
        db.commit()

    task.status = "exported"
    task.step = max(task.step, 7)
    job.status = "completed"
    job.progress = 100
    job.message = "本机导出完成"
    job.result_json = json.dumps(
        {"files": files, "demo": False, "run_key": run_key}, ensure_ascii=False
    )
    db.commit()


def _local_prelabel(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    """数据集 AI 预标注（短训 + 推理）；临时权重用完即删。"""
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds or not ds.path:
        raise FileNotFoundError("关联数据集不存在")
    root = Path(ds.path)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    if not isinstance(classes, list) or not classes:
        raise ValueError("请先添加至少一个类别")
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes
    dataset_storage.write_meta(root, meta)

    device = str(payload.get("device") or "cpu")

    def update_progress(progress: float, message: str) -> None:
        job.progress = float(progress)
        job.message = message
        db.commit()

    try:
        result = dataset_prelabel.run_prelabel_job(
            root=root,
            job_id=job.id,
            task_type=ds.task_type or "detect",
            device=device,
            update_progress=update_progress,
            is_cancelled=lambda: job_control.is_cancelled(job.id),
        )
        job.status = "completed"
        job.progress = 100
        job.message = result.get("message") or "预标注完成"
        job.result_json = json.dumps(result, ensure_ascii=False)
        db.commit()
    except InterruptedError:
        job.status = "cancelled"
        job.message = "预标注已取消"
        db.commit()
    finally:
        dataset_prelabel.cleanup_prelabel_tmp(root)
