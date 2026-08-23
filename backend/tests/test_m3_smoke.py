"""M3 冒烟：创建任务、划分、启动训练。"""

from __future__ import annotations

import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.core.database import init_db
from app.main import app
from app.services.bootstrap import ensure_storage_dirs

client = TestClient(app)


def main() -> None:
    init_db()
    ensure_storage_dirs()
    token = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"}).json()[
        "access_token"
    ]
    h = {"Authorization": f"Bearer {token}"}
    name_ds = f"m3_smoke_ds_{int(time.time())}"
    ds = client.post("/api/v1/datasets", headers=h, json={"name": name_ds}).json()
    dsid = ds["id"]
    files = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buf, "JPEG")
        files.append(("files", (f"a{i}.jpg", buf.getvalue(), "image/jpeg")))
    up = client.post(f"/api/v1/datasets/{dsid}/images", headers=h, files=files)
    assert up.status_code == 200, up.text

    name = f"smoke_task_{int(time.time())}"
    t = client.post("/api/v1/tasks", headers=h, json={"name": name, "dataset_id": dsid})
    assert t.status_code == 201, t.text
    tid = t.json()["id"]
    s = client.post(f"/api/v1/tasks/{tid}/split", headers=h)
    assert s.status_code == 200, s.text
    j = client.post(f"/api/v1/tasks/{tid}/train", headers=h)
    assert j.status_code == 200, j.text
    assert client.get("/api/v1/weights", headers=h).status_code == 200
    assert client.get("/api/v1/models", headers=h).status_code == 200
    print("SMOKE_OK", "task", tid, "job", j.json()["id"], "owner", t.json()["owner_id"])


if __name__ == "__main__":
    main()
