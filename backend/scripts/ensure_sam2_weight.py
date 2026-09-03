"""确保 pretrained/sam/sam2_b.pt 存在（缺失时从 Ultralytics assets 下载）。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# 允许从 backend/ 目录直接运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.sam_assist import SAM2_WEIGHT_NAME, ensure_sam_dir, resolve_sam2_weight  # noqa: E402


def main() -> int:
    dest = ensure_sam_dir() / SAM2_WEIGHT_NAME
    try:
        path = resolve_sam2_weight()
        print(f"已存在：{path} ({path.stat().st_size} bytes)")
        return 0
    except Exception:
        pass

    print(f"正在下载 {SAM2_WEIGHT_NAME} …")
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset(SAM2_WEIGHT_NAME, repo="ultralytics/assets", release="v8.3.0"))
    if downloaded.resolve() != dest.resolve():
        shutil.copy2(downloaded, dest)
        # 若下到 cwd，顺手清理
        if downloaded.name == SAM2_WEIGHT_NAME and downloaded.parent != dest.parent:
            try:
                downloaded.unlink()
            except OSError:
                pass
    print(f"完成：{dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
