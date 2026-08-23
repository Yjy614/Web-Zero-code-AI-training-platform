"""清理已无执行线程但仍标记为进行中的 Job（常见于服务重启）。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.train import Job, TrainTask
from app.runners import job_control


def close_orphan_job(db: Session, job: Job, message: str = "任务进程已结束，已自动标记为停止") -> bool:
    """
    若 Job 在库中为 pending/running，但本进程没有对应执行线程，则标记为 cancelled。
    返回是否做了修正。
    """
    if job.status not in ("pending", "running"):
        return False
    if job_control.is_job_active(job.id):
        return False
    job.status = "cancelled"
    job.message = message
    task = db.query(TrainTask).filter(TrainTask.id == job.task_id).first()
    if task and task.status == "training":
        # 有产出权重则保留 trained，否则回到 configured
        task.status = "trained" if task.model_path else "configured"
    db.commit()
    db.refresh(job)
    return True


def reconcile_orphan_jobs(db: Session) -> int:
    """启动时批量清理僵尸 Job；返回修正条数。"""
    rows = db.query(Job).filter(Job.status.in_(["pending", "running"])).all()
    n = 0
    for job in rows:
        if close_orphan_job(db, job, "服务已重启，原训练进程已中断"):
            n += 1
    return n
