"""Pose 主动学习策略：框置信度 + 关键点可见性。"""

from __future__ import annotations

from typing import Any

from app.schemas.dataset import Keypoint, PoseInstance
from app.services.active_learning.conf_score import WRITE_MIN_CONF, score_by_box_confidence
from app.services.active_learning.types import DetectionBox, ImageScore

# 可见关键点占比过低 → 难例
KPT_VISIBLE_EASY_RATIO = 0.55
KPT_CONF_VISIBLE = 0.5  # 推理 conf→v 时的阈值，与 model_infer 一致


class PoseActiveLearnStrategy:
    """姿态估计难易划分 + YOLO-Pose 写出。"""

    task_type = "pose"

    def score_image(
        self,
        *,
        image_name: str,
        detections: list[DetectionBox],
        image_size: tuple[int, int] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageScore:
        _ = image_size, extra
        base = score_by_box_confidence(
            image_name=image_name,
            detections=detections,
            empty_reason="未检出人体/目标，建议人工确认是否漏检",
            uncertain_reason="姿态框置信度不稳定，建议优先复核关键点",
            easy_reason="模型较有把握，已写姿态草稿，快速过一眼即可",
            low_reason="姿态框置信度偏低，建议人工确认",
        )
        if base.difficulty == "empty":
            return base

        vis_ratios: list[float] = []
        for d in detections:
            kpts = d.keypoints or []
            if not kpts:
                vis_ratios.append(0.0)
                continue
            visible = 0
            for kp in kpts:
                if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                    continue
                # v>=1：遮挡或可见均算有效关键点
                if float(kp[2]) >= 1:
                    visible += 1
            vis_ratios.append(visible / max(1, len(kpts)))

        mean_vis = sum(vis_ratios) / max(1, len(vis_ratios))
        if mean_vis < KPT_VISIBLE_EASY_RATIO:
            return ImageScore(
                image_name=image_name,
                difficulty="hard",
                score=max(base.score, round(1.0 - mean_vis, 4)),
                max_conf=base.max_conf,
                top2_gap=base.top2_gap,
                detection_count=len(detections),
                reason=f"关键点可见/有效比例偏低（{mean_vis:.0%}），建议优先复核",
                detections=detections,
            )
        return base

    def detections_to_label_payload(
        self,
        score: ImageScore,
        *,
        image_width: int,
        image_height: int,
    ) -> list[PoseInstance]:
        """像素框 + 关键点 → YOLO-Pose 归一化实例。"""
        w = max(1, int(image_width))
        h = max(1, int(image_height))
        out: list[PoseInstance] = []
        for d in score.detections:
            if float(d.confidence) < WRITE_MIN_CONF:
                continue
            x1, y1, x2, y2 = [float(v) for v in d.bbox_xyxy]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            kpts: list[Keypoint] = []
            for kp in d.keypoints or []:
                if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                    continue
                kx, ky = float(kp[0]), float(kp[1])
                raw_v = float(kp[2]) if len(kp) >= 3 else 2.0
                # 兼容：若传入的是 conf(0~1)，转成 YOLO v
                if 0.0 <= raw_v <= 1.0 and raw_v != 0 and raw_v != 1 and raw_v != 2:
                    v = 2 if raw_v >= KPT_CONF_VISIBLE else (1 if raw_v > 0.01 else 0)
                else:
                    v = int(raw_v)
                    if v < 0:
                        v = 0
                    if v > 2:
                        v = 2
                kpts.append(
                    Keypoint(
                        x=min(1.0, max(0.0, kx / w)),
                        y=min(1.0, max(0.0, ky / h)),
                        v=v,
                    )
                )
            out.append(
                PoseInstance(
                    class_id=int(d.class_id),
                    x_center=min(1.0, max(0.0, xc)),
                    y_center=min(1.0, max(0.0, yc)),
                    width=min(1.0, max(1e-6, bw / w)),
                    height=min(1.0, max(1e-6, bh / h)),
                    keypoints=kpts,
                )
            )
        return out
