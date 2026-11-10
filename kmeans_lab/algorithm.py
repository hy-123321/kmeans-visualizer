"""K-Means 核心迭代（状态机）。

维护人：袁恒宇（PM）

对外契约（其他模块只能依赖这些）：
    KMeans2D(points, k, seed=42, initialization="kmeans++",
             tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS)
属性：points / k / centroids / labels / inertia(即 SSE) / iteration / converged / history
方法：step() -> (旧质心, 新质心, 是否收敛)
辅助：assign_clusters(points, centroids) -> 每个样本的簇下标
"""
from __future__ import annotations

import math
import random

from .config import MAX_ITERATIONS, MAX_K, MIN_K, TOLERANCE, Point
from .initialization import initialize_centroids
from .metrics import inertia as calculate_inertia


def assign_clusters(points: list[Point], centroids: list[Point]) -> list[int]:
    """将每个样本分配给欧氏距离最近的质心。"""
    return [
        min(
            range(len(centroids)),
            key=lambda cluster: (
                (point[0] - centroids[cluster][0]) ** 2
                + (point[1] - centroids[cluster][1]) ** 2
            ),
        )
        for point in points
    ]


class KMeans2D:
    """K-means 状态机：初始化、单轮更新、收敛判断。"""

    def __init__(
        self,
        points: list[Point],
        k: int,
        seed: int = 42,
        initialization: str = "kmeans++",
        tolerance: float = TOLERANCE,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        try:
            self.points = [(float(point[0]), float(point[1])) for point in points]
            if any(len(point) != 2 for point in points):
                raise ValueError
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("输入数据必须是形状为 (样本数, 2) 的二维数组。") from error
        if not self.points or any(
            not math.isfinite(value) for point in self.points for value in point
        ):
            raise ValueError("输入数据必须全部为有限数值。")
        if not MIN_K <= k <= min(MAX_K, len(self.points)):
            raise ValueError(f"K 必须介于 {MIN_K} 和 {min(MAX_K, len(self.points))} 之间。")
        if initialization not in ("kmeans++", "random"):
            raise ValueError("初始化方式必须为 kmeans++ 或 random。")

        self.k = k
        self.rng = random.Random(seed)
        self.initialization = initialization
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration = 0
        self.converged = False
        self.history: list[float] = []
        self.centroids = initialize_centroids(
            self.points, k, self.rng, initialization
        )
        self.labels = assign_clusters(self.points, self.centroids)
        self.inertia = calculate_inertia(self.points, self.labels, self.centroids)
        self.history.append(self.inertia)

    # ---- 便捷别名：指标计算统一走 metrics.py，避免三处各写一遍 ----------
    def _calculate_inertia(self) -> float:
        return calculate_inertia(self.points, self.labels, self.centroids)

    def set_centroids(self, centroids: list[Point]) -> None:
        """手动放置质心时使用：直接接管质心并重算归属与 SSE。"""
        if len(centroids) != self.k:
            raise ValueError(f"需要 {self.k} 个质心，实际收到 {len(centroids)} 个。")
        self.centroids = [(float(x), float(y)) for x, y in centroids]
        self.labels = assign_clusters(self.points, self.centroids)
        self.inertia = self._calculate_inertia()
        self.history = [self.inertia]

    def step(self) -> tuple[list[Point], list[Point], bool]:
        """执行一轮“分配 + 更新”，返回旧质心、新质心、是否收敛。"""
        if self.converged or self.iteration >= self.max_iterations:
            return self.centroids.copy(), self.centroids.copy(), True

        old_centroids = self.centroids.copy()
        old_labels = assign_clusters(self.points, old_centroids)
        new_centroids = old_centroids.copy()
        distances_squared = [
            (point[0] - old_centroids[label][0]) ** 2
            + (point[1] - old_centroids[label][1]) ** 2
            for point, label in zip(self.points, old_labels)
        ]
        empty_clusters: list[int] = []

        for cluster in range(self.k):
            members = [
                point
                for point, label in zip(self.points, old_labels)
                if label == cluster
            ]
            if len(members):
                new_centroids[cluster] = (
                    sum(point[0] for point in members) / len(members),
                    sum(point[1] for point in members) / len(members),
                )
            else:
                empty_clusters.append(cluster)

        if empty_clusters:
            farthest_first = sorted(
                range(len(distances_squared)),
                key=distances_squared.__getitem__,
                reverse=True,
            )
            for cluster, point_index in zip(empty_clusters, farthest_first):
                new_centroids[cluster] = self.points[point_index]

        movement = [
            math.dist(new, old) for old, new in zip(old_centroids, new_centroids)
        ]
        self.centroids = new_centroids
        self.labels = assign_clusters(self.points, self.centroids)
        self.inertia = self._calculate_inertia()
        self.iteration += 1
        self.converged = (
            all(distance <= self.tolerance for distance in movement)
            or self.iteration >= self.max_iterations
        )
        self.history.append(self.inertia)
        return old_centroids, new_centroids.copy(), self.converged

    def run_to_convergence(self) -> None:
        """一口气跑到收敛（K 值分析用，不需要动画）。"""
        while not self.converged:
            self.step()
