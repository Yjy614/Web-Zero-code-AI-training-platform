"""数据集 API：CRUD、上传、清洗、标注。"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.dataset import Dataset
from app.models.train import Job, TrainTask
from app.models.user import User
from app.runners.mock_runner import get_runner
from app.schemas.dataset import (
    AnnotationOut,
    AnnotationPayload,
    ClassesUpdate,
    CleanResult,
    DatasetCreate,
    DatasetOut,
    ImageItem,
    ImageNamesPayload,
    PoseSkeletonOut,
    PoseSkeletonUpdate,
    SamAssistOut,
    SamAssistRequest,
)
from app.schemas.task import JobOut
from app.services import dataset_annotate, dataset_clean, dataset_prelabel, dataset_storage, sam_assist
from app.services.bootstrap import username_by_id
from app.services.runtime_settings import is_demo_mode
from app.core.task_types import normalize_task_type

router = APIRouter(prefix="/datasets", tags=["数据集"])


def _guess_image_media(path: Path) -> str:
    """按后缀给出图片 MIME，避免浏览器无法解码。"""
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _to_out(ds: Dataset) -> DatasetOut:
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    root = Path(ds.path) if ds.path else None
    image_count = ds.image_count
    active_count = 0
    if root is not None and root.exists():
        # 以磁盘为准：总数含排除项，活跃数为参与训练
        image_count = dataset_storage.count_images(root)
        active_count = dataset_storage.count_active_images(root)
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        path="",  # 不向前端暴露服务器绝对路径
        task_type=ds.task_type,
        owner_id=ds.owner_id,
        classes=classes if isinstance(classes, list) else [],
        image_count=image_count,
        active_count=active_count,
        created_at=ds.created_at,
        updated_at=ds.updated_at,
    )


def _get_accessible(db: Session, dataset_id: int, user: User) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "数据集不存在"})
    if user.role != "admin" and ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "无权访问该数据集"})
    return ds


def _refresh_count(db: Session, ds: Dataset) -> None:
    root = Path(ds.path)
    ds.image_count = dataset_storage.count_images(root) if root.exists() else 0
    db.commit()
    db.refresh(ds)


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    task_type: str = "detect",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DatasetOut]:
    """列出数据集（user 仅自己的，admin 全部）。"""
    try:
        tt = normalize_task_type(task_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_task_type", "message": str(e)}) from e
    from app.services.active_learning.staging import cleanup_orphaned_staging, is_active_learn_staging

    # 顺带清理悬空 AL 临时集（中断未并入的）
    try:
        cleanup_orphaned_staging(db, user)
    except Exception:  # noqa: BLE001
        pass

    q = db.query(Dataset).filter(Dataset.task_type == tt)
    if user.role != "admin":
        q = q.filter(Dataset.owner_id == user.id)
    rows = q.order_by(Dataset.id.desc()).all()
    # 主动学习临时集在并入确认前不出现在数据集管理
    return [_to_out(r) for r in rows if not is_active_learn_staging(r)]


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def create_dataset(
    body: DatasetCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetOut:
    """创建数据集目录与元数据（支持 detect / segment / pose）。"""
    try:
        name = dataset_storage.validate_dataset_name(body.name)
        tt = normalize_task_type(body.task_type)
    except ValueError as e:
        code = "bad_task_type" if "任务类型" in str(e) else "bad_name"
        raise HTTPException(status_code=400, detail={"code": code, "message": str(e)}) from e

    # 同一用户、同一任务类型下数据集名唯一
    conflict = (
        db.query(Dataset)
        .filter(Dataset.name == name, Dataset.task_type == tt, Dataset.owner_id == user.id)
        .first()
    )
    if conflict:
        raise HTTPException(status_code=400, detail={"code": "exists", "message": "您已有同名数据集"})

    root = dataset_storage.ensure_dataset_dirs(
        user.username, name, owner_id=user.id, task_type=tt
    )
    # 姿态：按创建参数覆盖默认 COCO-17
    if tt == "pose":
        from app.services.pose_skeleton import coco17_config, custom_config, normalize_pose_config

        try:
            tpl = (body.pose_template or "coco17").strip().lower()
            if tpl == "custom":
                pose_cfg = custom_config(
                    kpt_names=body.pose_kpt_names,
                    kpt_count=body.pose_kpt_count,
                    skeleton=body.pose_skeleton,
                )
            else:
                pose_cfg = coco17_config()
            pose_cfg = normalize_pose_config(pose_cfg)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"code": "bad_pose", "message": str(e)}) from e
        meta = dataset_storage.read_meta(root)
        meta["pose"] = pose_cfg
        meta["task_type"] = "pose"
        dataset_storage.write_meta(root, meta)

    ds = Dataset(
        name=name,
        path=str(root),
        task_type=tt,
        owner_id=user.id,
        classes_json="[]",
        image_count=0,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return _to_out(ds)


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetOut:
    """数据集详情。"""
    return _to_out(_get_accessible(db, dataset_id, user))


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """将数据集打包为 zip 下载（不含服务器绝对路径信息）。"""
    from fastapi.responses import StreamingResponse

    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    if not root.exists():
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "数据集目录不存在"})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            # 跳过划分副本可保留；全部打包便于用户带走
            arcname = file_path.relative_to(root).as_posix()
            zf.write(file_path, arcname)
    buf.seek(0)
    filename = f"{ds.name}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """删除数据集元数据与磁盘目录。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    db.delete(ds)
    db.commit()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    return {"message": "已删除数据集"}


@router.post("/{dataset_id}/images")
async def upload_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """多文件上传图片到 images/。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped: list[str] = []
    for f in files:
        raw_name = Path(f.filename or "image.jpg").name
        if not dataset_storage.is_safe_image_name(raw_name):
            skipped.append(raw_name)
            continue
        target = images_dir / raw_name
        # 重名则追加序号
        if target.exists():
            stem, suf = target.stem, target.suffix
            i = 1
            while True:
                candidate = images_dir / f"{stem}_{i}{suf}"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        content = await f.read()
        target.write_bytes(content)
        saved += 1

    _refresh_count(db, ds)
    return {"saved": saved, "skipped": skipped, "image_count": ds.image_count}


@router.post("/{dataset_id}/images/zip")
async def upload_zip(
    dataset_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """上传 zip，解压其中的图片到 images/。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    data = await file.read()
    saved = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not dataset_storage.is_safe_image_name(name):
                    continue
                target = images_dir / name
                if target.exists():
                    stem, suf = target.stem, target.suffix
                    i = 1
                    while (images_dir / f"{stem}_{i}{suf}").exists():
                        i += 1
                    target = images_dir / f"{stem}_{i}{suf}"
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                saved += 1
    except zipfile.BadZipFile as e:
        raise HTTPException(status_code=400, detail={"code": "bad_zip", "message": "无效的 ZIP 文件"}) from e

    _refresh_count(db, ds)
    return {"saved": saved, "image_count": ds.image_count}


@router.get("/{dataset_id}/images", response_model=list[ImageItem])
def list_dataset_images(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ImageItem]:
    """列出参与训练与不参与训练（排除）的图片。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    dataset_storage.migrate_physical_removed_to_excluded(root)
    excluded = dataset_storage.get_excluded(root)
    items: list[ImageItem] = []
    for name in dataset_storage.list_images(root, include_removed=True):
        has_label = dataset_storage.label_path_for(root, name).exists()
        status = "removed" if name in excluded else "active"
        items.append(ImageItem(name=name, has_label=has_label, status=status))
    return items


@router.get("/{dataset_id}/images/{image_name}/file")
def get_image_file(
    dataset_id: int,
    image_name: str,
    removed: bool = False,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """返回图片文件（标注页预览）。排除项仍在 images/，旧 removed/ 作回退。"""
    ds = _get_accessible(db, dataset_id, user)
    if not dataset_storage.is_safe_image_name(image_name):
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": "非法文件名"})
    root = Path(ds.path)
    path = root / "images" / image_name
    if not path.exists():
        path = root / "removed" / "images" / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "图片不存在"})
    return FileResponse(path, media_type=_guess_image_media(path))


@router.post("/{dataset_id}/clean", response_model=CleanResult)
def clean_dataset(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CleanResult:
    """一键清洗：长边归一 + 感知哈希去重（软排除，不删文件）。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    result = dataset_clean.clean_dataset(root)
    _refresh_count(db, ds)
    return CleanResult(**result)


@router.post("/{dataset_id}/restore")
def restore_dataset(
    dataset_id: int,
    body: ImageNamesPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """恢复已选中的排除图片，使其重新参与训练。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    result = dataset_clean.restore_selected(root, body.names)
    _refresh_count(db, ds)
    result["image_count"] = ds.image_count
    return result


@router.post("/{dataset_id}/images/delete")
def delete_dataset_images(
    dataset_id: int,
    body: ImageNamesPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """永久删除已选中图片（需用户确认）。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    result = dataset_clean.delete_selected(root, body.names)
    _refresh_count(db, ds)
    result["image_count"] = ds.image_count
    return result


def _ensure_task_for_dataset(db: Session, ds: Dataset, user: User) -> TrainTask:
    """预标注 Job 依赖 train_tasks：找不到则自动创建占位任务。"""
    task = (
        db.query(TrainTask)
        .filter(TrainTask.dataset_id == ds.id)
        .order_by(TrainTask.id.desc())
        .first()
    )
    if task:
        return task
    task = TrainTask(
        name=f"{ds.name}_train",
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        status="draft",
        step=2,
        config_json="{}",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _job_to_out(job: Job) -> JobOut:
    try:
        result = json.loads(job.result_json or "{}")
    except json.JSONDecodeError:
        result = {}
    if not isinstance(result, dict):
        result = {}
    return JobOut(
        id=job.id,
        task_id=job.task_id,
        type=job.type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result=result,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _pick_prelabel_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


@router.get("/{dataset_id}/prelabel/status")
def prelabel_status(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """预标注门槛与本轮可撤销信息。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes if isinstance(classes, list) else []
    dataset_storage.write_meta(root, meta)
    info = dataset_prelabel.analyze_prelabel(root, ds.task_type or "detect")
    reason = dataset_prelabel.prelabel_block_reason(info)
    task = (
        db.query(TrainTask)
        .filter(TrainTask.dataset_id == ds.id)
        .order_by(TrainTask.id.desc())
        .first()
    )
    running = None
    if task:
        running = (
            db.query(Job)
            .filter(
                Job.task_id == task.id,
                Job.type == "prelabel",
                Job.status.in_(["pending", "running"]),
            )
            .order_by(Job.id.desc())
            .first()
        )
    return {
        **{k: info[k] for k in (
            "labeled_count",
            "unlabeled_count",
            "class_count",
            "can_prelabel",
            "min_labeled",
            "min_unlabeled",
            "last_written",
        )},
        "block_reason": reason,
        "running_job_id": running.id if running else None,
    }


@router.post("/{dataset_id}/prelabel", response_model=JobOut)
async def start_prelabel(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobOut:
    """启动 AI 预标注（短训 + 推理未标注图）。"""
    ds = _get_accessible(db, dataset_id, user)
    root = Path(ds.path)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes if isinstance(classes, list) else []
    dataset_storage.write_meta(root, meta)

    info = dataset_prelabel.analyze_prelabel(root, ds.task_type or "detect")
    reason = dataset_prelabel.prelabel_block_reason(info)
    if reason:
        raise HTTPException(status_code=400, detail={"code": "prelabel_blocked", "message": reason})

    task = _ensure_task_for_dataset(db, ds, user)
    running = (
        db.query(Job)
        .filter(
            Job.task_id == task.id,
            Job.type == "prelabel",
            Job.status.in_(["pending", "running"]),
        )
        .first()
    )
    if running:
        raise HTTPException(status_code=400, detail={"code": "busy", "message": "已有预标注任务在进行中"})

    if not is_demo_mode():
        try:
            dataset_prelabel.resolve_prelabel_weight(ds.task_type or "detect")
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail={"code": "weight_missing", "message": str(e)}) from e
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "deps_missing",
                    "message": "未安装 ultralytics，请在后端环境执行：pip install ultralytics",
                },
            ) from e

    job = Job(task_id=task.id, type="prelabel", status="pending", progress=0, message="排队中")
    db.add(job)
    db.commit()
    db.refresh(job)

    await get_runner().submit(
        "prelabel",
        {
            "job_id": job.id,
            "task_id": task.id,
            "dataset_id": ds.id,
            "device": _pick_prelabel_device(),
            "owner": username_by_id(db, user.id),
        },
    )
    db.refresh(job)
    return _job_to_out(job)


@router.post("/{dataset_id}/prelabel/revert")
def revert_prelabel(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """清除本轮 AI 预标注写入的标注。"""
    ds = _get_accessible(db, dataset_id, user)
    result = dataset_prelabel.revert_prelabel(Path(ds.path))
    return result


@router.get("/{dataset_id}/classes")
def get_classes(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """获取类别列表。"""
    ds = _get_accessible(db, dataset_id, user)
    try:
        classes = json.loads(ds.classes_json or "[]")
    except json.JSONDecodeError:
        classes = []
    return {"classes": classes}


@router.put("/{dataset_id}/classes")
def put_classes(
    dataset_id: int,
    body: ClassesUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """更新类别列表，并同步 meta.json；若有删除则清理全库对应标注。"""
    ds = _get_accessible(db, dataset_id, user)
    classes = [c.strip() for c in body.classes if c and c.strip()]
    removed_indices = [int(i) for i in (body.removed_indices or []) if int(i) >= 0]
    ds.classes_json = json.dumps(classes, ensure_ascii=False)
    db.commit()
    root = Path(ds.path)
    meta = dataset_storage.read_meta(root)
    meta["classes"] = classes
    meta["name"] = ds.name
    meta["owner_id"] = ds.owner_id
    from app.services.bootstrap import username_by_id

    meta["owner_username"] = username_by_id(db, ds.owner_id)
    meta["task_type"] = ds.task_type
    dataset_storage.write_meta(root, meta)

    # 删除类别：去掉该 class_id 标注，并将更大 id 前移；再清掉越界孤儿
    purge_info = dataset_annotate.purge_and_remap_class_ids(
        root,
        task_type=ds.task_type or "detect",
        removed_indices=removed_indices,
        class_count=len(classes),
    )
    return {"classes": classes, "purge": purge_info}


@router.get("/{dataset_id}/pose-skeleton", response_model=PoseSkeletonOut)
def get_pose_skeleton(
    dataset_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PoseSkeletonOut:
    """读取姿态数据集的关键点骨架配置。"""
    ds = _get_accessible(db, dataset_id, user)
    if (ds.task_type or "detect") != "pose":
        raise HTTPException(
            status_code=400,
            detail={"code": "not_pose", "message": "仅姿态估计数据集支持骨架配置"},
        )
    from app.services.pose_skeleton import pose_config_from_meta

    cfg = pose_config_from_meta(dataset_storage.read_meta(Path(ds.path)))
    return PoseSkeletonOut.model_validate(cfg)


@router.put("/{dataset_id}/pose-skeleton", response_model=PoseSkeletonOut)
def put_pose_skeleton(
    dataset_id: int,
    body: PoseSkeletonUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PoseSkeletonOut:
    """更新姿态骨架（建议在标注前配置；已有标注时关键点数变化可能导致旧标签需重标）。"""
    ds = _get_accessible(db, dataset_id, user)
    if (ds.task_type or "detect") != "pose":
        raise HTTPException(
            status_code=400,
            detail={"code": "not_pose", "message": "仅姿态估计数据集支持骨架配置"},
        )
    from app.services.pose_skeleton import coco17_config, custom_config, normalize_pose_config

    try:
        tpl = (body.template or "coco17").strip().lower()
        if tpl == "custom":
            cfg = custom_config(
                kpt_names=body.kpt_names,
                kpt_count=body.kpt_count,
                skeleton=body.skeleton,
                flip_idx=body.flip_idx,
            )
        else:
            cfg = coco17_config()
        cfg = normalize_pose_config(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "bad_pose", "message": str(e)}) from e
    root = Path(ds.path)
    meta = dataset_storage.read_meta(root)
    meta["pose"] = cfg
    meta["task_type"] = "pose"
    dataset_storage.write_meta(root, meta)
    return PoseSkeletonOut.model_validate(cfg)


@router.get("/{dataset_id}/annotations/{image_name}", response_model=AnnotationOut)
def get_annotation(
    dataset_id: int,
    image_name: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnnotationOut:
    """读取单张图标注（检测 boxes / 分割 polygons / 姿态 poses）。"""
    ds = _get_accessible(db, dataset_id, user)
    if not dataset_storage.is_safe_image_name(image_name):
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": "非法文件名"})
    root = Path(ds.path)
    tt = ds.task_type or "detect"
    if tt == "segment":
        polygons = dataset_annotate.read_polygons(root, image_name)
        return AnnotationOut(image=image_name, boxes=[], polygons=polygons, poses=[])
    if tt == "pose":
        poses = dataset_annotate.read_poses(root, image_name)
        return AnnotationOut(image=image_name, boxes=[], polygons=[], poses=poses)
    boxes = dataset_annotate.read_annotations(root, image_name)
    return AnnotationOut(image=image_name, boxes=boxes, polygons=[], poses=[])


@router.put("/{dataset_id}/annotations/{image_name}", response_model=AnnotationOut)
def put_annotation(
    dataset_id: int,
    image_name: str,
    body: AnnotationPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnnotationOut:
    """保存单张图 YOLO 标注。"""
    ds = _get_accessible(db, dataset_id, user)
    if not dataset_storage.is_safe_image_name(image_name):
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": "非法文件名"})
    # 确认图片存在
    img_path = Path(ds.path) / "images" / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "图片不存在"})
    root = Path(ds.path)
    tt = ds.task_type or "detect"
    if tt == "segment":
        dataset_annotate.write_polygons(root, image_name, body.polygons)
        return AnnotationOut(image=image_name, boxes=[], polygons=body.polygons, poses=[])
    if tt == "pose":
        dataset_annotate.write_poses(root, image_name, body.poses)
        return AnnotationOut(image=image_name, boxes=[], polygons=[], poses=body.poses)
    dataset_annotate.write_annotations(root, image_name, body.boxes)
    return AnnotationOut(image=image_name, boxes=body.boxes, polygons=[], poses=[])

@router.post("/{dataset_id}/sam-assist", response_model=SamAssistOut)
def sam_assist_point(
    dataset_id: int,
    body: SamAssistRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SamAssistOut:
    """
    分割标注辅助：用 SAM2（sam2_b.pt）对图片上的一点做提示分割，
    返回归一化多边形，由前端写入当前类别。
    """
    ds = _get_accessible(db, dataset_id, user)
    if (ds.task_type or "detect") != "segment":
        raise HTTPException(
            status_code=400,
            detail={"code": "not_segment", "message": "仅实例分割数据集支持 SAM 辅助标注"},
        )
    if not dataset_storage.is_safe_image_name(body.image):
        raise HTTPException(status_code=400, detail={"code": "bad_name", "message": "非法文件名"})
    img_path = Path(ds.path) / "images" / body.image
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "图片不存在"})
    try:
        result = sam_assist.predict_polygon_from_point(
            img_path,
            x_norm=body.x,
            y_norm=body.y,
            positive=body.positive,
        )
    except sam_assist.SamAssistError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message}) from e
    return SamAssistOut(
        points=result["points"],
        model=str(result.get("model") or "sam2_b.pt"),
        width=int(result.get("width") or 0),
        height=int(result.get("height") or 0),
    )
