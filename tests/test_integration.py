"""联调与导出测试（负责人：叶世航）。

覆盖“数据 -> 迭代 -> 指标 -> 导出”整条链路，不依赖图形界面。
"""
from __future__ import annotations

import csv
import os
import tempfile
import unittest

from kmeans_lab.algorithm import KMeans2D
from kmeans_lab.data import generate_points
from kmeans_lab.export import CSV_HEADER, write_csv
from kmeans_lab.metrics import silhouette_score


class IntegrationTests(unittest.TestCase):
    def test_blobs_dataset_converges_to_three_clusters(self) -> None:
        points = generate_points("blobs", 180, 42, 0.65)
        model = KMeans2D(points, 3, seed=42)
        model.run_to_convergence()
        self.assertTrue(model.converged)
        self.assertEqual(len(set(model.labels)), 3)
        score = silhouette_score(points, model.labels)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.3)
        self.assertLess(model.history[-1], model.history[0])

    def test_write_csv_roundtrip(self) -> None:
        points = generate_points("blobs", 60, 3, 0.6)
        model = KMeans2D(points, 3, seed=3)
        model.run_to_convergence()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "result.csv")
            write_csv(path, points, model.labels, model.centroids)
            with open(path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], CSV_HEADER)
        self.assertEqual(len(rows) - 1, len(points))
        self.assertEqual(len(rows[1]), 5)
        cluster_ids = {int(row[2]) for row in rows[1:]}
        self.assertEqual(cluster_ids, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
