"""核心算法与初始化测试（负责人：袁恒宇，初始化部分由张迎奥补充）。"""
from __future__ import annotations

import math
import random
import unittest

from kmeans_lab.algorithm import KMeans2D, assign_clusters
from kmeans_lab.config import MAX_ITERATIONS
from kmeans_lab.initialization import (
    initialize_centroids,
    kmeans_plus_plus_centroids,
)


class AlgorithmTests(unittest.TestCase):
    def test_assign_clusters_picks_nearest(self) -> None:
        points = [(0.0, 0.0), (9.0, 9.0), (5.4, 5.0)]
        centroids = [(0.0, 0.0), (9.0, 9.0)]
        self.assertEqual(assign_clusters(points, centroids), [0, 1, 1])

    def test_initial_centroids_are_distinct_samples(self) -> None:
        points = [(float(index), float(index % 3)) for index in range(20)]
        for method in ("kmeans++", "random"):
            centroids = initialize_centroids(points, 4, random.Random(1), method)
            self.assertEqual(len(centroids), 4)
            self.assertEqual(len(set(centroids)), 4)
            self.assertTrue(all(center in points for center in centroids))

    def test_kmeans_plus_plus_reproducible(self) -> None:
        points = [(float(index % 7), float(index // 7)) for index in range(49)]
        first = kmeans_plus_plus_centroids(points, 3, random.Random(7))
        second = kmeans_plus_plus_centroids(points, 3, random.Random(7))
        self.assertEqual(first, second)

    def test_kmeans_converges_and_centers_are_means(self) -> None:
        first_rng = random.Random(2)
        second_rng = random.Random(3)
        points = [
            (first_rng.gauss(2.0, 0.2), first_rng.gauss(2.0, 0.2)) for _ in range(50)
        ]
        points.extend(
            (second_rng.gauss(8.0, 0.2), second_rng.gauss(8.0, 0.2))
            for _ in range(50)
        )
        model = KMeans2D(points, 2, seed=5)
        for _ in range(MAX_ITERATIONS):
            model.step()
            if model.converged:
                break
        self.assertTrue(model.converged)
        self.assertEqual(model.iteration, len(model.history) - 1)
        for cluster in range(model.k):
            members = [
                point for point, label in zip(points, model.labels) if label == cluster
            ]
            self.assertGreater(len(members), 0)
            self.assertAlmostEqual(
                model.centroids[cluster][0],
                sum(point[0] for point in members) / len(members),
            )
            self.assertAlmostEqual(
                model.centroids[cluster][1],
                sum(point[1] for point in members) / len(members),
            )
        self.assertLess(model.inertia, model.history[0])

    def test_invalid_k_raises(self) -> None:
        points = [(0.0, 0.0), (0.2, 0.0), (5.0, 5.0), (5.2, 5.0)]
        with self.assertRaises(ValueError):
            KMeans2D(points, 1)

    def test_duplicate_points_and_empty_clusters_are_safe(self) -> None:
        points = [(1.0, 2.0)] * 12
        model = KMeans2D(points, 3, seed=7)
        model.step()
        self.assertTrue(model.converged)
        self.assertEqual(model.inertia, 0.0)
        self.assertTrue(
            all(
                math.isfinite(coordinate)
                for centroid in model.centroids
                for coordinate in centroid
            )
        )

    def test_set_centroids_recomputes_labels(self) -> None:
        points = [(0.0, 0.0), (0.5, 0.0), (9.0, 9.0), (9.5, 9.0)]
        model = KMeans2D(points, 2, seed=1)
        model.set_centroids([(0.0, 0.0), (9.0, 9.0)])
        self.assertEqual(model.labels, [0, 0, 1, 1])
        self.assertLess(model.inertia, 1.0)
        with self.assertRaises(ValueError):
            model.set_centroids([(0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
