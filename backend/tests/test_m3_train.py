"""M3 联调：划分 + Mock 训练/评估/导出（缩短等待可单独测 API）。"""

from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.core.database import init_db
from app.main import app
from app.services.bootstrap import ensure_storage_dirs

client = TestClient(app)


def main() -> None:
    init_db()
    ensure_storage_dirs()
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}

    # 准备数据集
    name = "m3_demo_ds"
    for d in client.get("/api/v1/datasets", headers=h).json():
        if d["name"] == name:
            client.delete(f"/api/v1/datasets/{d['id']}", headers=h)

    ds = client.post("/api/v1/datasets", headers=h, json={"name": name}).json()
    dsid = ds["id"]
    files = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        buf = io.BytesIO()
        Image.new("RGB", (120, 80), color).save(buf, format="JPEG")
        files.append(("files", (f"img{i}.jpg", buf.getvalue(), "image/jpeg")))
    client.post(f"/api/v1/datasets/{dsid}/images", headers=h, files=files)
    client.put(f"/api/v1/datasets/{dsid}/classes", headers=h, json={"classes": ["目标"]})

    # 任务
    tname = "m3_demo_task"
    for t in client.get("/api/v1/tasks", headers=h).json():
        if t["name"] == tname:
            # 无删除 API，跳过占用：换名
            tname = f"m3_demo_task_{int(time.time())}"
            break

    task = client.post("/api/v1/tasks", headers=h, json={"name": tname, "dataset_id": dsid}).json()
    tid = task["id"]
    client.patch(
        f"/api/v1/tasks/{tid}",
        headers=h,
        json={"config": {"train_ratio": 0.75, "val_ratio": 0.25, "epochs": 20, "batch": 4, "imgsz": 640, "device": "cpu", "pretrained_weight": "yolov8n.pt", "augment": True}},
    )
    split = client.post(f"/api/v1/tasks/{tid}/split", headers=h)
    print("split", split.status_code, split.json())
    assert split.status_code == 200

    job = client.post(f"/api/v1/tasks/{tid}/train", headers=h).json()
    print("train job", job["id"])
    # 轮询至完成（Mock 约数十秒）
    for _ in range(80):
        j = client.get(f"/api/v1/jobs/{job['id']}", headers=h).json()
        if j["status"] in {"completed", "failed", "cancelled"}:
            print("train done", j["status"], j["message"], "progress", j["progress"])
            break
        time.sleep(1)
    else:
        raise AssertionError("训练超时")
    assert j["status"] == "completed"

    task = client.get(f"/api/v1/tasks/{tid}", headers=h).json()
    mp_parts = Path(task["model_path"]).parts
    assert "/runs/" in task["model_path"].replace("\\", "/") or "runs" in mp_parts
    assert not any(p.startswith("user_") and p[5:].isdigit() for p in mp_parts)
    print("model_path", task["model_path"])

    ej = client.post(f"/api/v1/tasks/{tid}/eval", headers=h).json()
    for _ in range(30):
        j = client.get(f"/api/v1/jobs/{ej['id']}", headers=h).json()
        if j["status"] in {"completed", "failed", "cancelled"}:
            print("eval done", j["status"])
            break
        time.sleep(0.5)
    assert j["status"] == "completed"

    xj = client.post(f"/api/v1/tasks/{tid}/export", headers=h, json={"formats": ["pt", "onnx"]}).json()
    for _ in range(20):
        j = client.get(f"/api/v1/jobs/{xj['id']}", headers=h).json()
        if j["status"] in {"completed", "failed", "cancelled"}:
            print("export done", j["result"])
            break
        time.sleep(0.5)
    assert j["status"] == "completed"

    models = client.get("/api/v1/models", headers=h).json()
    print("models", len(models))
    assert any(m.get("task_id") == tid for m in models)

    root = Path(task["model_path"]).parent.parent  # runs/.../weights -> runs/.../
    print("runs exists", root.exists())
    print("OK")


if __name__ == "__main__":
    main()
