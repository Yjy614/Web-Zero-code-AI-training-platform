"""训练任务与 Job API。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.dataset import Dataset
from app.models.train import Job, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner, request_cancel
from app.runners import job_control
from app.schemas.task import AiAdviceOut, ExportRequest, JobOut, TaskCreate, TaskOut, TaskUpdate, TrainConfig
from app.services import task_storage, yolo_split
from app.services.ai_advice import generate_ai_advice
from app.services.bootstrap import username_by_id
from app.services.eval_report import write_eval_reports
from app.services.job_reconcile import close_orphan_job
from app.services.runtime_settings import is_demo_mode
from app.services.weight_resolve import normalize_weight_name, resolve_pretrained_weight

router = APIRouter(tags=["训练任务"])


def _parse_json(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _task_type(db: Session, task: TrainTask) -> str:
    """从关联数据集解析任务类型（detect / segment），默认 detect。"""
    ds = db.query(Dataset).filter(Dataset.id == task.dataset_id).first()
    return (ds.task_type if ds and ds.task_type else None) or "detect"

def _task_out(task: TrainTask) -> TaskOut:
    cfg = _parse_json(task.config_json)
    # 读出时清洗历史占位权重名，避免前端下拉仍显示无效值
    pw = normalize_weight_name(str(cfg.get("pretrained_weight") or ""))
    if cfg.get("pretrained_weight") != pw:
        cfg["pretrained_weight"] = pw
    return TaskOut(
        id=task.id,
        name=task.name,
        owner_id=task.owner_id,
        dataset_id=task.dataset_id,
        status=task.status,
        step=task.step,
        config=cfg,
        model_path=task.model_path or "",
        metrics=_parse_json(task.metrics_json),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        task_id=job.task_id,
        type=job.type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=_parse_json(job.result_json),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _get_task(db: Session, task_id: int, user: User) -> TrainTask:
    task = db.query(TrainTask).filter(TrainTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "训练任务不存在"})
    if user.role != "admin" and task.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问该任务"})
    return task


def _get_dataset(db: Session, dataset_id: int, user: User) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "数据集不存在"})
    if user.role != "admin" and ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权使用该数据集"})
    return ds


@router.get("/tasks", response_model=list[TaskOut])
def list_tasks(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[TaskOut]:
    q = db.query(TrainTask)
    if user.role != "admin":
        q = q.filter(TrainTask.owner_id == user.id)
    return [_task_out(t) for t in q.order_by(TrainTask.id.desc()).all()]


@router.post("/tasks", response_model=TaskOut, status_code=status.HTTP_201_CREATED)
def create_task(body: TaskCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TaskOut:
    try:
        name = task_storage.validate_task_name(body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": str(e)}) from e

    _get_dataset(db, body.dataset_id, user)
    ds = db.query(Dataset).filter(Dataset.id == body.dataset_id).first()
    task_type = (ds.task_type if ds else None) or "detect"
    exists = (
        db.query(TrainTask)
        .filter(TrainTask.name == name, TrainTask.owner_id == user.id)
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail={"code": "exists", "message": "您已有同名训练任务"})

    task_storage.ensure_task_dirs(user.username, name, task_type)
    cfg = TrainConfig().model_dump()
    task = TrainTask(
        name=name,
        owner_id=user.id,
        dataset_id=body.dataset_id,
        status="draft",
        step=0,
        config_json=json.dumps(cfg, ensure_ascii=False),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return _task_out(task)


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TaskOut:
    return _task_out(_get_task(db, task_id, user))


@router.patch("/tasks/{task_id}", response_model=TaskOut)
def patch_task(
    task_id: int,
    body: TaskUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TaskOut:
    task = _get_task(db, task_id, user)
    if body.step is not None:
        task.step = body.step
    if body.status is not None:
        task.status = body.status
    if body.config is not None:
        task.config_json = json.dumps(body.config.model_dump(), ensure_ascii=False)
    db.commit()
    db.refresh(task)
    return _task_out(task)


@router.post("/tasks/{task_id}/split")
def split_task(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    """按配置划分数据集并写 data.yaml。"""
    task = _get_task(db, task_id, user)
    ds = _get_dataset(db, task.dataset_id, user)
    cfg = TrainConfig(**_parse_json(task.config_json))
    try:
        result = yolo_split.prepare_yolo_split(
            Path(ds.path),
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "split_failed", "message": str(e)}) from e
    task.status = "configured"
    task.step = max(task.step, 4)
    db.commit()
    return {"message": "划分完成", **result}


@router.post("/tasks/{task_id}/train", response_model=JobOut)
async def start_train(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> JobOut:
    task = _get_task(db, task_id, user)
    cfg = TrainConfig(**_parse_json(task.config_json))
    # 若尚未划分，自动划分一次
    ds = _get_dataset(db, task.dataset_id, user)
    if not (Path(ds.path) / "data.yaml").exists():
        try:
            yolo_split.prepare_yolo_split(
                Path(ds.path),
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"code": "split_failed", "message": str(e)}) from e

    running = (
        db.query(Job)
        .filter(Job.task_id == task.id, Job.type == "train", Job.status.in_(["pending", "running"]))
        .first()
    )
    if running:
        raise HTTPException(status_code=400, detail={"code": "busy", "message": "已有训练任务在进行中"})

    # 真实训练前校验本地权重是否存在（按数据集任务类型分目录）
    if not is_demo_mode():
        tt = ds.task_type or "detect"
        try:
            resolve_pretrained_weight(cfg.pretrained_weight, tt, db=db)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail={"code": "weight_missing", "message": str(e)}) from e
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "deps_missing",
                    "message": "未安装 ultralytics，请在后端环境执行：pip install ultralytics",
                },
            ) from e

    job = Job(task_id=task.id, type="train", status="pending", progress=0, message="排队中")
    db.add(job)
    db.commit()
    db.refresh(job)

    runner = get_runner()
    await runner.submit(
        "train",
        {"job_id": job.id, "epochs": cfg.epochs, "task_id": task.id},
    )
    db.refresh(job)
    return _job_out(job)


@router.post("/tasks/{task_id}/eval", response_model=JobOut)
async def start_eval(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> JobOut:
    task = _get_task(db, task_id, user)
    if task.status not in {"trained", "evaluated", "exported"} and not task.model_path:
        raise HTTPException(status_code=400, detail={"code": "not_ready", "message": "请先完成训练"})

    job = Job(task_id=task.id, type="eval", status="pending", progress=0, message="排队中")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit("eval", {"job_id": job.id, "task_id": task.id})
    db.refresh(job)
    return _job_out(job)


@router.post("/tasks/{task_id}/ai-advice", response_model=AiAdviceOut)
def task_ai_advice(task_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> AiAdviceOut:
    """基于数据集规模与可调参数，调用设置中的大模型生成优化建议，并写入评估报告。"""
    task = _get_task(db, task_id, user)
    ds = _get_dataset(db, task.dataset_id, user)
    result = generate_ai_advice(task, ds)
    advice = str(result.get("advice") or "")
    source = str(result.get("source") or "fallback")

    metrics = _parse_json(task.metrics_json)
    metrics["ai_advice"] = advice
    metrics["ai_advice_source"] = source
    task.metrics_json = json.dumps(metrics, ensure_ascii=False)

    # 同步刷新 HTML 报告，便于下载包含 AI 建议
    tt = _task_type(db, task)
    run_key = task_storage.resolve_run_key(task_name=task.name, model_path=task.model_path)
    # metrics 里若已有 run_key 优先（与本轮评估一致）
    mk = str(metrics.get("run_key") or "").strip()
    if mk:
        run_key = mk
    report_dir = task_storage.versioned_reports_dir(
        username_by_id(db, task.owner_id), task.name, run_key, tt
    )
    write_eval_reports(task.name, metrics, report_dir)
    db.commit()
    db.refresh(task)

    return AiAdviceOut(
        advice=advice,
        source=source,
        context=result.get("context") if isinstance(result.get("context"), dict) else {},
        error=result.get("error"),
    )


@router.post("/tasks/{task_id}/export", response_model=JobOut)
async def start_export(
    task_id: int,
    body: ExportRequest | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    task = _get_task(db, task_id, user)
    if task.status not in {"trained", "evaluated", "exported"} and not task.model_path:
        raise HTTPException(status_code=400, detail={"code": "not_ready", "message": "请先完成训练"})
    formats = (body.formats if body else None) or ["pt", "onnx"]

    job = Job(task_id=task.id, type="export", status="pending", progress=0, message="排队中")
    db.add(job)
    db.commit()
    db.refresh(job)
    await get_runner().submit("export", {"job_id": job.id, "task_id": task.id, "formats": formats})
    db.refresh(job)
    return _job_out(job)


@router.get("/tasks/{task_id}/jobs", response_model=list[JobOut])
def list_task_jobs(
    task_id: int,
    type: str | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    """列出某训练任务下的 Job（新到旧），用于前端恢复进行中的训练/评估。"""
    task = _get_task(db, task_id, user)
    q = db.query(Job).filter(Job.task_id == task.id)
    if type:
        q = q.filter(Job.type == type)
    rows = q.order_by(Job.id.desc()).limit(30).all()
    out: list[JobOut] = []
    for job in rows:
        # 清理因重启等残留的「假运行」状态，避免前端误恢复
        close_orphan_job(db, job)
        out.append(_job_out(job))
    return out


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> JobOut:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Job 不存在"})
    _get_task(db, job.task_id, user)
    close_orphan_job(db, job)
    return _job_out(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Job 不存在"})
    _get_task(db, job.task_id, user)
    if job.status not in ("pending", "running"):
        return {"message": "任务已结束，无需停止"}
    request_cancel(job_id)
    # 进程里已无执行线程（例如服务重启后）：直接标为已停止
    if not job_control.is_job_active(job.id):
        close_orphan_job(db, job, "训练已停止")
        return {"message": "训练已停止"}
    if job.status == "running":
        job.message = "正在停止：将在当前 epoch 结束后停止，请稍候…"
        db.commit()
    return {"message": "已请求停止，将在当前 epoch 结束后停止"}


@router.get("/tasks/{task_id}/download/{kind}")
def download_artifact(
    task_id: int,
    kind: str,
    name: str | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """下载报告或导出文件。kind=report|export|weight"""
    task = _get_task(db, task_id, user)
    if kind == "weight":
        path = Path(task.model_path) if task.model_path else None
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "权重不存在"})
        return FileResponse(path, filename=path.name)

    if kind == "report":
        tt = _task_type(db, task)
        username = username_by_id(db, task.owner_id)
        run_key = task_storage.resolve_run_key(task_name=task.name, model_path=task.model_path)
        metrics_preview = _parse_json(task.metrics_json)
        if str(metrics_preview.get("run_key") or "").strip():
            run_key = str(metrics_preview.get("run_key")).strip()
        report_dir = task_storage.versioned_reports_dir(username, task.name, run_key, tt)
        # 兼容旧版扁平报告
        legacy_report_dir = task_storage.reports_dir(username, task.name, tt)
        file_name = name or "eval_report.html"
        try:
            path = task_storage.safe_join_under(report_dir, Path(file_name).name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"code": "bad_name", "message": str(e)}) from e
        # 下载前按最新 metrics（含 AI 建议）重写 HTML
        if path.suffix.lower() in {".html", ".htm"} or file_name == "eval_report.html":
            metrics = _parse_json(task.metrics_json)
            json_path = report_dir / "eval_report.json"
            if not json_path.exists():
                json_path = legacy_report_dir / "eval_report.json"
            if json_path.exists():
                try:
                    disk = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(disk, dict):
                        merged = {**disk, **metrics}
                        if metrics.get("ai_advice"):
                            merged["ai_advice"] = metrics["ai_advice"]
                            merged["ai_advice_source"] = metrics.get("ai_advice_source")
                        metrics = merged
                except (json.JSONDecodeError, OSError):
                    pass
            write_eval_reports(task.name, metrics, report_dir)
            path = report_dir / "eval_report.html"
        if not path.exists():
            legacy = legacy_report_dir / Path(file_name).name
            if legacy.exists():
                path = legacy
            else:
                raise HTTPException(status_code=404, detail={"code": "not_found", "message": "报告不存在"})
        return FileResponse(path, filename=path.name)

    if kind == "export":
        tt = _task_type(db, task)
        username = username_by_id(db, task.owner_id)
        export_root = task_storage.exports_dir(username, task.name, tt)
        if not name:
            raise HTTPException(status_code=400, detail={"code": "bad_request", "message": "请指定文件名"})
        # 新版：run_key/filename；旧版：扁平 filename
        path: Path | None = None
        try:
            path = task_storage.safe_join_under(export_root, name)
        except ValueError:
            path = None
        if path is None or not path.is_file():
            flat = export_root / Path(name).name
            if flat.is_file():
                path = flat
            else:
                # 再试：任意版本子目录下的同名文件（取最近修改）
                matches = sorted(
                    export_root.glob(f"*/{Path(name).name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                path = matches[0] if matches else None
        if not path or not path.is_file():
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "导出文件不存在"})
        return FileResponse(path, filename=path.name)

    raise HTTPException(status_code=400, detail={"code": "bad_kind", "message": "不支持的下载类型"})
