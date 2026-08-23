"""训练任务取消标记与存活登记（Mock / Local 共用）。"""

from __future__ import annotations

import threading

_cancel_flags: dict[int, bool] = {}
_active_jobs: set[int] = set()
_lock = threading.Lock()


def clear_cancel(job_id: int) -> None:
    with _lock:
        _cancel_flags[job_id] = False


def request_cancel(job_id: int) -> None:
    with _lock:
        _cancel_flags[job_id] = True


def is_cancelled(job_id: int) -> bool:
    with _lock:
        return bool(_cancel_flags.get(job_id, False))


def mark_job_active(job_id: int) -> None:
    """线程开始执行时登记，供判断 Job 是否仍在本进程运行。"""
    with _lock:
        _active_jobs.add(job_id)


def mark_job_inactive(job_id: int) -> None:
    with _lock:
        _active_jobs.discard(job_id)


def is_job_active(job_id: int) -> bool:
    with _lock:
        return job_id in _active_jobs
