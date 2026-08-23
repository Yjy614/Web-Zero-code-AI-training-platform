"""数据库会话与初始化。"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
# SQLite 需关闭同线程检查，便于 FastAPI 多线程场景
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """ORM 基类。"""


def get_db() -> Generator[Session, None, None]:
    """依赖注入：请求级数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """创建表结构。"""
    # 延迟导入模型，确保元数据注册
    from app.models import dataset as _dataset  # noqa: F401
    from app.models import train as _train  # noqa: F401
    from app.models import user as _user  # noqa: F401

    Base.metadata.create_all(bind=engine)
