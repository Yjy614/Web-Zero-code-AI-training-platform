"""认证路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.security import create_access_token, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, UserOut

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """用户登录，返回 JWT。"""
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "bad_credentials", "message": "用户名或密码错误"},
        )
    token = create_access_token(user.username, {"role": user.role, "uid": user.id})
    return LoginResponse(access_token=token, user=UserOut.model_validate(user))


@router.post("/logout")
def logout(_: User = Depends(get_current_user)) -> dict:
    """登出（前端删除 Token 即可；服务端返回成功）。"""
    return {"message": "已退出登录"}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    """当前登录用户信息。"""
    return user
