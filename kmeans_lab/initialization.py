"""质心初始化：随机选点 与 K-means++。

维护人：张迎奥（算法工程师）

两种方式都必须返回 **k 个互不相同的样本** 作为初始质心，
且完全由传入的 `random.Random` 实例决定随机性（保证可复现）。
"""
from __future__ import annotations

import random

from .config import Point


def random_centroids(
    points: list[Point], k: int, rng: random.Random
) -> list[Point]:
    """从样本中不重复随机抽取 k 个点作为初始质心。"""
    indices = rng.sample(range(len(points)), k)
    return [points[index] for index in indices]


def kmeans_plus_plus_centroids(
    points: list[Point], k: int, rng: random.Random
) -> list[Point]:
    """K-means++：后续质心按 D(x)^2 概率抽样，让初始质心尽量分散。"""
    count = len(points)
    indices = [rng.randrange(count)]
    while len(indices) < k:
        distances_squared = [
            min(
                (point[0] - points[index][0]) ** 2
                + (point[1] - points[index][1]) ** 2
                for index in indices
            )
            for point in points
        ]
        for index in indices:
            distances_squared[index] = 0.0
        total = sum(distances_squared)
        remaining = [index for index in range(count) if index not in indices]
        if total <= 0.0:
            next_index = rng.choice(remaining)
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            next_index = count - 1
            for index, distance_squared in enumerate(distances_squared):
                cumulative += distance_squared
                if cumulative > threshold:
                    next_index = index
                    break
            if next_index >= count or next_index in indices:
                next_index = remaining[-1]
        indices.append(next_index)
    return [points[index] for index in indices]


def initialize_centroids(
    points: list[Point],
    k: int,
    rng: random.Random,
    method: str = "kmeans++",
) -> list[Point]:
    """统一入口：method 取 "kmeans++" 或 "random"。"""
    if method == "random":
        return random_centroids(points, k, rng)
    if method == "kmeans++":
        return kmeans_plus_plus_centroids(points, k, rng)
    raise ValueError("初始化方式必须为 kmeans++ 或 random。")
