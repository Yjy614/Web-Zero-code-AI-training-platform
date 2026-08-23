"""JobRunner：Mock 实现（演示期默认）。"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.train import Job, ModelRecord, TrainTask
from app.runners.base import ClusterJobRunner, JobRunner
from app.runners import job_control
from app.services import dataset_prelabel, task_storage
from app.services.bootstrap import username_by_id
from app.services.runtime_settings import is_demo_mode


class MockJobRunner(JobRunner):
    """演示用 Mock：假进度 + 假曲线 + 占位产物，不占 GPU。"""

    async def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        job_id = int(payload["job_id"])
        job_control.clear_cancel(job_id)
        # 后台线程跑同步循环，避免阻塞事件循环
        t = threading.Thread(target=_run_job_sync, args=(job_id, job_type, payload), daemon=True)
        t.start()
        return str(job_id)

    async def cancel(self, job_id: str) -> None:
        job_control.request_cancel(int(job_id))


def get_runner() -> JobRunner:
    """
    按演示开关与配置返回执行器：
    - demo_mode=true → Mock
    - job_runner=cluster → Cluster（未实现）
    - 否则 → Local（本机 Ultralytics）
    """
    settings = get_settings()
    if is_demo_mode():
        return MockJobRunner()
    if settings.job_runner == "cluster":
        return ClusterJobRunner()
    # 关闭演示后默认本机真实训练
    from app.runners.local_runner import LocalJobRunner

    return LocalJobRunner()


def request_cancel(job_id: int) -> None:
    job_control.request_cancel(job_id)


def _cancelled(job_id: int) -> bool:
    return job_control.is_cancelled(job_id)


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
            _mock_train(db, job, task, payload)
        elif job_type == "eval":
            _mock_eval(db, job, task, payload)
        elif job_type == "export":
            _mock_export(db, job, task, payload)
        elif job_type == "prelabel":
            _mock_prelabel(db, job, task, payload)
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
            db.commit()
    finally:
        job_control.mark_job_inactive(job_id)
        db.close()


def _mock_train(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    # 优先使用任务配置中的 epochs，与向导设置一致
    try:
        cfg = json.loads(task.config_json or "{}")
        if not isinstance(cfg, dict):
            cfg = {}
    except json.JSONDecodeError:
        cfg = {}
    epochs = int(payload.get("epochs") or cfg.get("epochs") or 50)
    steps = max(1, min(epochs, 500))
    # 演示时长控制在约 8~25 秒，不因 epochs 过大拖太久
    sleep_s = min(0.8, max(0.12, 18.0 / steps))
    history: list[dict] = []
    dirs = task_storage.ensure_task_dirs(username_by_id(db, task.owner_id), task.name)
    from app.services import model_artifacts

    archive_dir = task_storage.saved_models_dir(username_by_id(db, task.owner_id), "detect")
    model_name = model_artifacts.next_unique_model_name(
        db, task.owner_id, task.name, archive_dir=archive_dir
    )
    run_dir = dirs["runs"] / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_best = run_dir / "weights" / "best.pt"
    weights_best.parent.mkdir(parents=True, exist_ok=True)

    task.status = "training"
    db.commit()

    for i in range(1, steps + 1):
        if _cancelled(job.id):
            job.status = "cancelled"
            job.message = "训练已取消"
            task.status = "configured"
            db.commit()
            return
        import time

        time.sleep(sleep_s)
        progress = round(i / steps * 100, 1)
        loss = round(2.2 * math.exp(-i / (steps / 3)) + 0.15 + (0.03 * math.sin(i)), 4)
        map50 = round(min(0.92, 0.25 + i / steps * 0.65), 4)
        history.append({"epoch": i, "loss": loss, "map50": map50})
        job.progress = progress
        job.message = f"训练中 epoch {i}/{steps}"
        job.result_json = json.dumps({"history": history, "epochs": steps}, ensure_ascii=False)
        db.commit()

    # 写占位权重
    weights_best.write_bytes(b"MOCK_YOLO_WEIGHTS_DEMO_FILE\n")
    task.model_path = str(weights_best)
    task.status = "trained"
    task.step = max(task.step, 5)
    task.metrics_json = json.dumps(
        {"best_map50": history[-1]["map50"] if history else 0, "history": history, "epochs": steps},
        ensure_ascii=False,
    )

    # 登记模型库（每次训练新增，名称冲突用 _2、_3…）
    from app.services import model_artifacts

    username = username_by_id(db, task.owner_id)
    dest_dir = task_storage.saved_models_dir(username, "detect")
    dest_dir.mkdir(parents=True, exist_ok=True)
    archived = dest_dir / f"{model_name}.pt"
    if archived.exists():
        model_name = model_artifacts.next_unique_model_name(
            db, task.owner_id, task.name, archive_dir=dest_dir
        )
        archived = dest_dir / f"{model_name}.pt"
    archived.write_bytes(weights_best.read_bytes())
    metrics = {
        "map50": history[-1]["map50"] if history else 0,
        "loss": history[-1]["loss"] if history else 0,
        "demo": True,
        "epochs": steps,
    }
    db.add(
        ModelRecord(
            name=model_name,
            path=str(archived),
            task_type="detect",
            owner_id=task.owner_id,
            task_id=task.id,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
        )
    )

    # 训练成功：本轮 AI 预标注视为用户标注（保留框，仅去掉可撤销标记）
    _commit_prelabel_after_train(db, task)

    job.status = "completed"
    job.progress = 100
    job.message = f"Mock 训练完成（演示权重，{steps} epochs）"
    job.result_json = json.dumps(
        {"history": history, "model_path": str(weights_best), "demo": True, "epochs": steps},
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
        pass


def _mock_eval(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    import time

    dirs = task_storage.ensure_task_dirs(username_by_id(db, task.owner_id), task.name)
    report_dir = dirs["reports"]
    for i in range(1, 11):
        if _cancelled(job.id):
            job.status = "cancelled"
            job.message = "评估已取消"
            db.commit()
            return
        time.sleep(0.4)
        job.progress = i * 10
        job.message = f"评估中 {job.progress:.0f}%"
        db.commit()

    metrics = {
        "map50": 0.86,
        "map50_95": 0.61,
        "precision": 0.88,
        "recall": 0.83,
        "demo": True,
        "suggestion": "演示模式建议：增加难例样本、检查标注一致性，真实训练请关闭演示模式并配置集群。",
    }
    prev = {}
    try:
        prev = json.loads(task.metrics_json or "{}")
    except json.JSONDecodeError:
        prev = {}
    # 保留此前已生成的 AI 建议
    if prev.get("ai_advice") and not metrics.get("ai_advice"):
        metrics["ai_advice"] = prev.get("ai_advice")
        metrics["ai_advice_source"] = prev.get("ai_advice_source")

    from app.services.eval_report import write_eval_reports

    report_json, report_html = write_eval_reports(task.name, metrics, report_dir)

    task.status = "evaluated"
    task.step = max(task.step, 6)
    prev.update(metrics)
    task.metrics_json = json.dumps(prev, ensure_ascii=False)

    job.status = "completed"
    job.progress = 100
    job.message = "Mock 评估完成"
    job.result_json = json.dumps(
        {
            "metrics": metrics,
            "report_json": str(report_json),
            "report_html": str(report_html),
            "demo": True,
        },
        ensure_ascii=False,
    )
    db.commit()


def _mock_export(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    import time

    formats = payload.get("formats") or ["pt", "onnx"]
    dirs = task_storage.ensure_task_dirs(username_by_id(db, task.owner_id), task.name)
    export_dir = dirs["exports"]
    files = []
    total = max(len(formats), 1)
    for idx, fmt in enumerate(formats, start=1):
        if _cancelled(job.id):
            job.status = "cancelled"
            job.message = "导出已取消"
            db.commit()
            return
        time.sleep(0.5)
        out = export_dir / f"{task.name}_demo.{fmt}"
        out.write_bytes(f"MOCK_EXPORT_{fmt.upper()}_DEMO\n".encode("utf-8"))
        files.append({"format": fmt, "path": str(out), "name": out.name, "demo": True})
        job.progress = round(idx / total * 100, 1)
        job.message = f"正在导出 {fmt}"
        db.commit()

    task.status = "exported"
    task.step = max(task.step, 7)
    job.status = "completed"
    job.progress = 100
    job.message = "Mock 导出完成（演示文件）"
    job.result_json = json.dumps({"files": files, "demo": True}, ensure_ascii=False)
    db.commit()


def _mock_prelabel(db, job: Job, task: TrainTask, payload: dict[str, Any]) -> None:
    """演示模式预标注。"""
    from app.models.dataset import Dataset
    from app.services import dataset_prelabel, dataset_storage

    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    if not ds or not ds.path:
        raise FileNotFoundError("关联数据集不存在")
    root = Path(ds.path)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes if isinstance(classes, list) else []
    dataset_storage.write_meta(root, meta)

    def update_progress(progress: float, message: str) -> None:
        job.progress = float(progress)
        job.message = message
        db.commit()

    try:
        result = dataset_prelabel.run_prelabel_mock(
            root=root,
            job_id=job.id,
            update_progress=update_progress,
            is_cancelled=lambda: _cancelled(job.id),
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
