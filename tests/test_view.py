"""视图层测试：置信椭圆几何（负责人：庞照阳，测试维护：叶世航）。

只测纯几何函数，不启动 Tk 窗口。
"""
from __future__ import annotations

import math
import unittest

from kmeans_lab.view import confidence_ellipse


class ViewTests(unittest.TestCase):
    def test_ellipse_is_finite_and_closed(self) -> None:
        points = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
        ellipse = confidence_ellipse(points)
        self.assertEqual(len(ellipse), 65)
        self.assertEqual(ellipse[0], ellipse[-1])
        self.assertTrue(
            all(math.isfinite(value) for point in ellipse for value in point)
        )

    def test_ellipse_needs_two_points(self) -> None:
        self.assertEqual(confidence_ellipse([]), [])
        self.assertEqual(confidence_ellipse([(1.0, 1.0)]), [])

    def test_ellipse_contains_center(self) -> None:
        points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]
        ellipse = confidence_ellipse(points)
        xs = [point[0] for point in ellipse]
        ys = [point[1] for point in ellipse]
        # 椭圆点集应环绕均值 (1, 1)
        self.assertLess(min(xs), 1.0)
        self.assertGreater(max(xs), 1.0)
        self.assertLess(min(ys), 1.0)
        self.assertGreater(max(ys), 1.0)


if __name__ == "__main__":
    unittest.main()
