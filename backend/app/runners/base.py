"""JobRunner 抽象：一期 Mock，后期 Cluster。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class JobRunner(ABC):
    """任务执行器接口。"""

    @abstractmethod
    async def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        """提交任务，返回 job_id。"""

    @abstractmethod
    async def cancel(self, job_id: str) -> None:
        """取消任务。"""


class MockJobRunner(JobRunner):
    """演示用 Mock：不占用 GPU，仅占位。"""

    async def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        # M3 再实现进度推送与假曲线
        raise NotImplementedError("MockJobRunner 将在 M3 接入训练/评估/导出")

    async def cancel(self, job_id: str) -> None:
        return None


class ClusterJobRunner(JobRunner):
    """工厂 GPU 集群调度占位（后期实现）。"""

    async def submit(self, job_type: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError("ClusterJobRunner 尚未实现")

    async def cancel(self, job_id: str) -> None:
        raise NotImplementedError("ClusterJobRunner 尚未实现")
