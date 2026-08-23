"""M2 联调脚本：创建 / 上传 / 清洗 / 标注。"""

from __future__ import annotations

import io
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

    name = "m2_demo_ds"
    for d in client.get("/api/v1/datasets", headers=h).json():
        if d["name"] == name:
            client.delete(f"/api/v1/datasets/{d['id']}", headers=h)

    r = client.post("/api/v1/datasets", headers=h, json={"name": name})
    print("create", r.status_code, r.json())
    assert r.status_code == 201, r.text
    ds = r.json()
    dsid = ds["id"]

    files = []
    for fname, color, size in [
        ("a.jpg", (255, 0, 0), (200, 100)),
        ("b.jpg", (250, 5, 5), (200, 100)),
        ("c.jpg", (0, 0, 255), (80, 80)),
    ]:
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="JPEG")
        files.append(("files", (fname, buf.getvalue(), "image/jpeg")))

    r = client.post(f"/api/v1/datasets/{dsid}/images", headers=h, files=files)
    print("upload", r.status_code, r.json())
    assert r.status_code == 200, r.text

    r = client.post(f"/api/v1/datasets/{dsid}/clean", headers=h)
    print("clean", r.status_code, r.json())
    assert r.status_code == 200, r.text

    r = client.put(f"/api/v1/datasets/{dsid}/classes", headers=h, json={"classes": ["缺陷", "划痕"]})
    print("classes", r.json())

    imgs = client.get(f"/api/v1/datasets/{dsid}/images", headers=h).json()
    active = [i for i in imgs if i["status"] == "active"]
    print("images", imgs)
    assert active, "应至少保留一张图"
    imgname = active[0]["name"]

    r = client.put(
        f"/api/v1/datasets/{dsid}/annotations/{imgname}",
        headers=h,
        json={"boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.3}]},
    )
    print("anno", r.status_code, r.json())
    assert r.status_code == 200, r.text

    root = Path(ds["path"])
    from app.core.database import SessionLocal
    from app.models.user import User

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == ds["owner_id"]).first()
        assert u is not None
        assert u.username in root.parts, f"路径应按用户名隔离: {root}"
        assert f"user_{ds['owner_id']}" not in root.parts
    finally:
        db.close()
    print("path", root)
    print("disk images", list((root / "images").glob("*")))
    print("disk labels", list((root / "labels").glob("*")))
    print("OK")


if __name__ == "__main__":
    main()
