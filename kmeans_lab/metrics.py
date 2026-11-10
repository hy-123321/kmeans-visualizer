"""聚类质量指标：SSE（簇内平方和）、轮廓系数、K 值推荐。

维护人：张迎奥（算法工程师）

约定：
* `inertia(...)` 就是界面上的“簇内平方和 (SSE)”，值越小簇越紧。
* `silhouette_score(...)` 返回 [-1, 1] 的平均轮廓系数；无法计算时返回 None。
* `recommend_k_values(...)` 输入 [(K, SSE, silhouette or None), ...] 的结果表，
  返回 (肘部 K, 轮廓峰值 K 或 None, 峰值或 None)。
"""
from __future__ import annotations

import math

from .config import Point


def squared_distance(a: Point, b: Point) -> float:
    """两点欧氏距离的平方。"""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def inertia(points: list[Point], labels: list[int], centroids: list[Point]) -> float:
    """SSE：每个样本到其所属质心的距离平方之和。"""
    return sum(
        squared_distance(point, centroids[label])
        for point, label in zip(points, labels)
    )


def silhouette_score(points: list[Point], labels: list[int]) -> float | None:
    """计算平均轮廓系数；无有效多簇划分时返回 None。"""
    clusters = sorted(set(labels))
    if len(clusters) < 2 or len(clusters) >= len(points):
        return None

    members = {
        cluster: [index for index, label in enumerate(labels) if label == cluster]
        for cluster in clusters
    }
    scores = []
    for index, label in enumerate(labels):
        own_members = members[label]
        own_count = len(own_members)
        if own_count <= 1:
            scores.append(0.0)
            continue
        within = sum(
            math.dist(points[index], points[member])
            for member in own_members
            if member != index
        ) / (own_count - 1)
        nearest_other = min(
            sum(math.dist(points[index], points[member]) for member in other_members)
            / len(other_members)
            for cluster, other_members in members.items()
            if cluster != label
        )
        denominator = max(within, nearest_other)
        scores.append(
            0.0 if denominator <= 0.0 else (nearest_other - within) / denominator
        )
    return sum(scores) / len(scores)


def recommend_k_values(
    results: list[tuple[int, float, float | None]],
) -> tuple[int, int | None, float | None]:
    """根据 SSE 曲线最大弦距（肘部）和轮廓系数峰值推荐 K。"""
    k_values = [result[0] for result in results]
    inertias = [result[1] for result in results]
    inertia_range = max(inertias) - min(inertias)
    if len(results) < 3 or inertia_range <= 0.0:
        elbow_k = k_values[0]
    else:
        x_range = k_values[-1] - k_values[0]
        normalized_x = [(k - k_values[0]) / x_range for k in k_values]
        normalized_sse = [
            (value - min(inertias)) / inertia_range for value in inertias
        ]
        distances = [
            abs(y - (1.0 - x)) / math.sqrt(2.0)
            for x, y in zip(normalized_x, normalized_sse)
        ]
        elbow_k = k_values[max(range(len(distances)), key=distances.__getitem__)]

    valid_silhouettes = [
        (result[0], result[2]) for result in results if result[2] is not None
    ]
    if not valid_silhouettes:
        return elbow_k, None, None
    silhouette_k, silhouette_value = max(
        valid_silhouettes, key=lambda result: result[1]
    )
    return elbow_k, silhouette_k, silhouette_value


def relative_reduction(initial: float, current: float) -> float:
    """SSE 相对初始状态的下降比例，用于状态栏文案。"""
    if initial <= 0.0:
        return 0.0
    return max(0.0, (initial - current) / initial)
