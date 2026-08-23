"""数据集清洗：长边归一 + 感知哈希去重（软排除，不物理删除）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.services.dataset_storage import (
    get_excluded,
    is_safe_image_name,
    label_path_for,
    list_images,
    migrate_physical_removed_to_excluded,
    set_excluded,
)

# 长边归一目标尺寸
LONG_EDGE = 1280
# 感知哈希汉明距离阈值（越小越严格）
PHASH_THRESHOLD = 5


def _average_hash(img: Image.Image, hash_size: int = 8) -> int:
    """简易平均哈希（不额外依赖 imagehash）。"""
    gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    return bits


def _color_bucket(img: Image.Image) -> tuple[int, int, int]:
    """主色分桶，避免纯色图因结构哈希全相同而误删。"""
    r, g, b = img.convert("RGB").resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0))
    return (r // 16, g // 16, b // 16)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _is_duplicate(sig_a: tuple[int, tuple[int, int, int]], sig_b: tuple[int, tuple[int, int, int]]) -> bool:
    """哈希接近且主色相近才视为重复。"""
    ha, ca = sig_a
    hb, cb = sig_b
    if ca != cb:
        return False
    return _hamming(ha, hb) <= PHASH_THRESHOLD


def _normalize_long_edge(path: Path) -> None:
    """将图片长边缩放到 LONG_EDGE（更小则保持）。"""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= LONG_EDGE:
            return
        scale = LONG_EDGE / float(long_edge)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        save_kwargs = {}
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 92
            save_kwargs["optimize"] = True
        resized.save(path, **save_kwargs)


def clean_dataset(root: Path) -> dict:
    """
    执行清洗：
    1. 长边归一
    2. 感知哈希去重：重复/损坏图仅写入 excluded，文件仍留在 images/
    """
    migrate_physical_removed_to_excluded(root)
    images = list_images(root, include_removed=False)
    excluded = get_excluded(root)
    kept_sigs: list[tuple[str, tuple[int, tuple[int, int, int]]]] = []
    newly_excluded: list[str] = []

    for name in images:
        img_path = root / "images" / name
        try:
            _normalize_long_edge(img_path)
            with Image.open(img_path) as img:
                sig = (_average_hash(img), _color_bucket(img))
        except Exception:
            excluded.add(name)
            newly_excluded.append(name)
            continue

        dup = any(_is_duplicate(sig, prev) for _, prev in kept_sigs)
        if dup:
            excluded.add(name)
            newly_excluded.append(name)
        else:
            kept_sigs.append((name, sig))

    set_excluded(root, excluded)
    return {
        "kept": len(kept_sigs),
        "removed": len(newly_excluded),
        "removed_files": newly_excluded,
        "message": (
            f"清洗完成：保留 {len(kept_sigs)} 张参与训练，"
            f"本轮排除 {len(newly_excluded)} 张（未从数据集删除）"
        ),
    }


def restore_selected(root: Path, names: list[str]) -> dict:
    """将选中的排除项恢复为参与训练（文件本就在 images/）。"""
    migrate_physical_removed_to_excluded(root)
    excluded = get_excluded(root)
    restored = 0
    for name in names:
        if not is_safe_image_name(name):
            continue
        if name in excluded:
            excluded.discard(name)
            restored += 1
    set_excluded(root, excluded)
    if restored == 0:
        return {"restored": 0, "message": "没有可恢复的已选图片"}
    return {"restored": restored, "message": f"已恢复 {restored} 张图片参与训练"}


def restore_removed(root: Path) -> dict:
    """兼容旧接口：恢复全部排除项。"""
    migrate_physical_removed_to_excluded(root)
    excluded = get_excluded(root)
    n = len(excluded)
    set_excluded(root, set())
    if n == 0:
        return {"restored": 0, "message": "没有可恢复的图片"}
    return {"restored": n, "message": f"已恢复 {n} 张图片参与训练"}


def delete_selected(root: Path, names: list[str]) -> dict:
    """用户确认后永久删除选中图片及标注。"""
    migrate_physical_removed_to_excluded(root)
    excluded = get_excluded(root)
    deleted = 0
    for name in names:
        if not is_safe_image_name(name):
            continue
        for img in (root / "images" / name, root / "removed" / "images" / name):
            if img.exists():
                img.unlink(missing_ok=True)
        for label in (
            label_path_for(root, name, removed=False),
            label_path_for(root, name, removed=True),
        ):
            if label.exists():
                label.unlink(missing_ok=True)
        excluded.discard(name)
        deleted += 1
    set_excluded(root, excluded)
    if deleted == 0:
        return {"deleted": 0, "message": "没有可删除的已选图片"}
    return {"deleted": deleted, "message": f"已永久删除 {deleted} 张图片"}
