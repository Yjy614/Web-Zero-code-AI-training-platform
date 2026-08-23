"""用户管理（管理员）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.core.database import get_db
from app.core.security import hash_password
from app.models.user import User
from app.schemas.auth import UserOut

router = APIRouter(prefix="/users", tags=["用户管理"])


class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=4, max_length=128)
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=4, max_length=128)
    role: str | None = Field(default=None, pattern="^(admin|user)$")


@router.get("", response_model=list[UserOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[User]:
    """列出全部用户。"""
    return db.query(User).order_by(User.id.asc()).all()


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> User:
    """创建用户。"""
    exists = db.query(User).filter(User.username == body.username).first()
    if exists:
        raise HTTPException(status_code=400, detail={"code": "user_exists", "message": "用户名已存在"})
    user = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    body: UserUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> User:
    """更新用户密码或角色。"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "用户不存在"})
    if body.role is not None and body.role != user.role:
        # 禁止取消最后一个管理员
        if user.role == "admin" and body.role != "admin":
            admin_count = db.query(User).filter(User.role == "admin").count()
            if admin_count <= 1:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "last_admin", "message": "不能取消最后一个管理员的角色"},
                )
        user.role = body.role
    if body.password:
        user.password_hash = hash_password(body.password)
    db.commit()
    db.refresh(user)
    return user


@router.post("/{user_id}/reset-password", response_model=UserOut)
def reset_password(
    user_id: int,
    body: UserUpdate,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> User:
    """重置密码（管理员）。"""
    if not body.password:
        raise HTTPException(status_code=400, detail={"code": "bad_request", "message": "请提供新密码"})
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "用户不存在"})
    user.password_hash = hash_password(body.password)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}")
def delete_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    """删除用户。"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "用户不存在"})
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail={"code": "cannot_delete_self", "message": "不能删除当前登录账号"})
    if user.role == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail={"code": "last_admin", "message": "不能删除最后一个管理员"})
    db.delete(user)
    db.commit()
    return {"message": "已删除"}
