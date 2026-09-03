"""FastAPI 应用入口。"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import agent, auth, datasets, resources, settings, tasks, users
from app.core.database import SessionLocal, init_db
from app.services.bootstrap import (
    ensure_storage_dirs,
    migrate_datasets_to_user_scope,
    repair_model_record_paths,
    seed_users,
)
from app.services.job_reconcile import reconcile_orphan_jobs


class _QuietPollAccessFilter(logging.Filter):
    """过滤训练进度轮询等访问日志，避免控制台被刷屏。"""

    _skip = re.compile(
        r'"GET /api/v1/(?:'
        r'jobs/\d+'
        r'|tasks/\d+(?:/jobs)?'
        r'|health'
        r'|datasets/\d+/images/[^"\s]+/file'
        r'|datasets/\d+/prelabel/status'
        r')(?:\?[^"]*)? HTTP/'
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return self._skip.search(msg) is None


def _install_quiet_access_filter() -> None:
    logger = logging.getLogger("uvicorn.access")
    # 避免重复挂载
    if any(isinstance(f, _QuietPollAccessFilter) for f in logger.filters):
        return
    logger.addFilter(_QuietPollAccessFilter())


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化数据库、目录与种子账号。"""
    _install_quiet_access_filter()
    init_db()
    ensure_storage_dirs()
    db = SessionLocal()
    try:
        seed_users(db)
        migrate_datasets_to_user_scope(db)
        repair_model_record_paths(db)
        reconcile_orphan_jobs(db)
    finally:
        db.close()
    yield


# 模块导入时也挂一次（兼容部分启动顺序）
_install_quiet_access_filter()

app = FastAPI(title="工业场景零代码 AI 训练平台", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    """统一未捕获异常格式（尽量中文提示）。"""
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": f"服务器内部错误：{exc}"},
    )


app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(settings.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(resources.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")


@app.get("/api/v1/health")
def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}
