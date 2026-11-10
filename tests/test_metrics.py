"""指标测试：SSE / 轮廓系数 / K 值推荐（负责人：张迎奥）。"""
from __future__ import annotations

import unittest

from kmeans_lab.metrics import (
    inertia,
    recommend_k_values,
    relative_reduction,
    silhouette_score,
)


class MetricsTests(unittest.TestCase):
    def test_inertia_matches_hand_calculation(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0)]
        centroids = [(1.0, 0.0), (1.0, 0.0)]
        # 每个点距离质心 1，SSE = 1 + 1 = 2
        self.assertAlmostEqual(inertia(points, [0, 1], centroids), 2.0)

    def test_silhouette_and_invalid_k(self) -> None:
        points = [(0.0, 0.0), (0.2, 0.0), (5.0, 5.0), (5.2, 5.0)]
        score = silhouette_score(points, [0, 0, 1, 1])
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)
        # 只有一个簇时无法计算轮廓系数
        self.assertIsNone(silhouette_score(points, [0, 0, 0, 0]))

    def test_k_recommendations_find_elbow_and_silhouette_peak(self) -> None:
        results = [
            (2, 100.0, 0.42),
            (3, 55.0, 0.61),
            (4, 40.0, 0.82),
            (5, 34.0, 0.73),
            (6, 30.0, 0.68),
            (7, 27.0, 0.59),
            (8, 25.0, 0.51),
        ]
        self.assertEqual(recommend_k_values(results), (4, 4, 0.82))

    def test_recommend_without_silhouette(self) -> None:
        results = [(2, 100.0, None), (3, 50.0, None), (4, 40.0, None)]
        elbow_k, silhouette_k, silhouette_value = recommend_k_values(results)
        self.assertIn(elbow_k, (2, 3, 4))
        self.assertIsNone(silhouette_k)
        self.assertIsNone(silhouette_value)

    def test_relative_reduction_is_clamped(self) -> None:
        self.assertAlmostEqual(relative_reduction(100.0, 40.0), 0.6)
        self.assertEqual(relative_reduction(0.0, 10.0), 0.0)
        self.assertEqual(relative_reduction(100.0, 120.0), 0.0)


if __name__ == "__main__":
    unittest.main()
