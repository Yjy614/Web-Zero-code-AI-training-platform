"""M4 冒烟：用户管理与设置。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.database import init_db
from app.main import app
from app.services.bootstrap import ensure_storage_dirs

client = TestClient(app)


def main() -> None:
    init_db()
    ensure_storage_dirs()
    admin = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"}).json()
    h = {"Authorization": f"Bearer {admin['access_token']}"}

    # 创用户
    r = client.post("/api/v1/users", headers=h, json={"username": "m4_user", "password": "pass1234", "role": "user"})
    assert r.status_code == 201, r.text
    uid = r.json()["id"]

    # 重置密码
    r = client.post(f"/api/v1/users/{uid}/reset-password", headers=h, json={"password": "newpass1"})
    assert r.status_code == 200, r.text

    # 用户登录
    u = client.post("/api/v1/auth/login", json={"username": "m4_user", "password": "newpass1"})
    assert u.status_code == 200, u.text
    uh = {"Authorization": f"Bearer {u.json()['access_token']}"}

    # 普通用户不能管用户
    assert client.get("/api/v1/users", headers=uh).status_code == 403

    # 设置：用户可读，不可写
    assert client.get("/api/v1/settings", headers=uh).status_code == 200
    assert client.put("/api/v1/settings", headers=uh, json={"demo_mode": False}).status_code == 403

    # 管理员写 LLM
    r = client.put(
        "/api/v1/settings",
        headers=h,
        json={"llm": {"base_url": "http://127.0.0.1:8080", "api_key": "sk-test-key", "model": "demo", "timeout": 30}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["llm"]["api_key"].startswith("••••")
    assert r.json()["llm"]["base_url"].endswith("8080")

    # 信息接口不暴露绝对路径字段
    info = client.get("/api/v1/system/info", headers=h).json()
    assert "storage_root" not in info
    assert info.get("storage_configured") is True

    # 清理
    client.delete(f"/api/v1/users/{uid}", headers=h)
    print("M4_SMOKE_OK")


if __name__ == "__main__":
    main()
