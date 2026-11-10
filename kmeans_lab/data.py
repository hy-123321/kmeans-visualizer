"""二维数据生成：三簇 / 四簇高斯、同心圆、笑脸轮廓。

维护人：张迎奥（算法工程师）

对外只暴露 `generate_points(dataset, count, seed, spread)`：
同样的 (dataset, count, seed, spread) 必须得到完全相同的数据，
这是叶世航的测试和讲解截图能复现的前提。
"""
from __future__ import annotations

import math
import random

from .config import DEFAULT_COUNT, Point


def generate_points(
    dataset: str,
    count: int = DEFAULT_COUNT,
    seed: int = 42,
    spread: float = 0.65,
) -> list[Point]:
    """按指定形状生成可复现的二维数据，坐标统一压缩在 [0, 10] 内。"""
    if count < 2:
        raise ValueError("数据点数量至少为 2。")
    rng = random.Random(seed)

    if dataset in ("blobs", "four_blobs"):
        centers: tuple[Point, ...] = (
            ((2.5, 3.0), (7.4, 3.0), (5.0, 7.2))
            if dataset == "blobs"
            else ((2.7, 2.8), (7.3, 2.8), (2.7, 7.2), (7.3, 7.2))
        )
        center_x = sum(center[0] for center in centers) / len(centers)
        center_y = sum(center[1] for center in centers) / len(centers)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        scale = rng.uniform(0.8, 1.2)
        cosine, sine = math.cos(angle), math.sin(angle)
        varied_centers = []
        for x, y in centers:
            x -= center_x
            y -= center_y
            varied_centers.append(
                (
                    center_x + scale * (x * cosine - y * sine)
                    + rng.uniform(-1.0, 1.0) + rng.gauss(0.0, spread * 1.1),
                    center_y + scale * (x * sine + y * cosine)
                    + rng.uniform(-1.0, 1.0) + rng.gauss(0.0, spread * 1.1),
                )
            )
        points = [
            (
                rng.gauss(center[0], spread),
                rng.gauss(center[1], spread),
            )
            for center in (
                varied_centers[rng.randrange(len(varied_centers))]
                for _ in range(count)
            )
        ]
    elif dataset == "rings":
        points = []
        for _ in range(count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = max(
                0.1,
                (1.8 if rng.randrange(2) == 0 else 3.8)
                + rng.gauss(0.0, spread * 0.22),
            )
            points.append(
                (5.0 + radius * math.cos(angle), 5.0 + radius * math.sin(angle))
            )
    elif dataset == "smiley":
        outline_count = count * 3 // 10
        eye_count = count * 3 // 10
        mouth_count = count - outline_count - eye_count
        points = []

        for _ in range(outline_count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.gauss(4.0, spread * 0.12)
            points.append(
                (5.0 + radius * math.cos(angle), 5.0 + radius * math.sin(angle))
            )

        left_eye_count = eye_count // 2
        right_eye_count = eye_count - left_eye_count
        for center_x, eye_count_for_side in ((3.5, left_eye_count), (6.5, right_eye_count)):
            points.extend(
                (
                    rng.gauss(center_x, spread * 0.38),
                    rng.gauss(6.5, spread * 0.48),
                )
                for _ in range(eye_count_for_side)
            )

        for _ in range(mouth_count):
            mouth_x = rng.uniform(2.6, 7.4)
            mouth_y = 3.2 + 0.2 * (mouth_x - 5.0) ** 2
            points.append(
                (
                    mouth_x + rng.gauss(0.0, spread * 0.12),
                    mouth_y + rng.gauss(0.0, spread * 0.15),
                )
            )
    else:
        raise ValueError(f"不支持的数据集：{dataset}")

    angle = rng.uniform(0.0, 2.0 * math.pi)
    scale = rng.uniform(0.9, 1.1)
    cosine, sine = math.cos(angle), math.sin(angle)
    offset_x, offset_y = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
    return [
        (
            min(
                10.0,
                max(
                    0.0,
                    5.0 + scale * ((x - 5.0) * cosine - (y - 5.0) * sine)
                    + offset_x,
                ),
            ),
            min(
                10.0,
                max(
                    0.0,
                    5.0 + scale * ((x - 5.0) * sine + (y - 5.0) * cosine)
                    + offset_y,
                ),
            ),
        )
        for x, y in points
    ]
