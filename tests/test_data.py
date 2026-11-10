"""数据生成测试（负责人：张迎奥）。"""
from __future__ import annotations

import math
import unittest

from kmeans_lab.config import DATASETS
from kmeans_lab.data import generate_points


class DataTests(unittest.TestCase):
    def test_generators_are_reproducible_and_finite(self) -> None:
        for dataset in DATASETS.values():
            first = generate_points(dataset, 120, 19, 0.6)
            second = generate_points(dataset, 120, 19, 0.6)
            self.assertEqual(len(first), 120)
            self.assertTrue(
                all(
                    math.isfinite(coordinate)
                    for point in first
                    for coordinate in point
                )
            )
            self.assertEqual(first, second)

    def test_points_stay_inside_canvas(self) -> None:
        for dataset in DATASETS.values():
            for x, y in generate_points(dataset, 80, 7, 1.3):
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 10.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 10.0)

    def test_blob_cluster_positions_vary_with_seed(self) -> None:
        first = generate_points("blobs", 180, 42, 0.65)
        second = generate_points("blobs", 180, 43, 0.65)
        first_mean = tuple(
            sum(point[axis] for point in first) / len(first) for axis in (0, 1)
        )
        second_mean = tuple(
            sum(point[axis] for point in second) / len(second) for axis in (0, 1)
        )
        self.assertGreater(math.dist(first_mean, second_mean), 0.5)

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_points("blobs", count=1)
        with self.assertRaises(ValueError):
            generate_points("not-a-dataset", count=10)


if __name__ == "__main__":
    unittest.main()
