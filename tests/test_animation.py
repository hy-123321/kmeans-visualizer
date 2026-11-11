"""动画控制测试（负责人：叶世航）。

用假 root 替代 Tkinter，手动推进定时器，因此不需要图形环境也能测动画循环。
覆盖：单步补间帧数、收敛后不再前进、手动放置质心时暂停、停止时取消定时器、连续运行链路。
"""
from __future__ import annotations

import unittest

from kmeans_lab.algorithm import KMeans2D
from kmeans_lab.animation import AnimationController
from kmeans_lab.data import generate_points


class FakeRoot:
    """记录 after / after_cancel，由测试手动推进。"""

    def __init__(self) -> None:
        self.queue: list[tuple[int, str, object]] = []
        self.cancelled: list[str] = []
        self._counter = 0

    def after(self, delay_ms: int, func):
        self._counter += 1
        identifier = f"after#{self._counter}"
        self.queue.append((delay_ms, identifier, func))
        return identifier

    def after_cancel(self, identifier: str) -> None:
        self.cancelled.append(identifier)
        self.queue = [item for item in self.queue if item[1] != identifier]

    def run_all(self) -> int:
        steps = 0
        while self.queue:
            _, _, func = self.queue.pop(0)
            func()
            steps += 1
            if steps > 500:  # 防止死循环
                raise AssertionError("定时器回调没有停止，可能形成死循环")
        return steps


class FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class FakeApp:
    """满足 AnimationController 所需的最小接口。"""

    def __init__(self, points, k: int = 3, seed: int = 1) -> None:
        self.root = FakeRoot()
        self.model = KMeans2D(points, k, seed=seed)
        self.placing_centroids = False
        self.status_var = FakeVar()
        self.draws: list[dict] = []

    def _draw(self, display_centroids=None, movement=None, refresh_metrics=True) -> None:
        self.draws.append(
            {
                "centroids": display_centroids,
                "movement": movement,
                "refresh": refresh_metrics,
            }
        )


def make_points():
    return generate_points("blobs", 120, 42, 0.65)


class AnimationTests(unittest.TestCase):
    def test_step_once_draws_every_frame(self) -> None:
        app = FakeApp(make_points())
        controller = AnimationController(app, frames=8, frame_ms=24)
        controller.step_once()
        self.assertTrue(controller.animating)
        self.assertEqual(len(app.draws), 1)  # 第一帧立即画出

        app.root.run_all()
        self.assertFalse(controller.animating)
        # 8 帧补间 + 收尾再画一次（刷新指标、去掉箭头）
        self.assertEqual(len(app.draws), 9)
        self.assertEqual(app.model.iteration, 1)
        # 补间帧只画移动、不刷新指标；收尾那一次才刷新
        self.assertIsNotNone(app.draws[0]["movement"])
        self.assertFalse(app.draws[0]["refresh"])
        self.assertIsNone(app.draws[-1]["movement"])
        self.assertTrue(app.draws[-1]["refresh"])

    def test_step_once_ignores_converged_model(self) -> None:
        app = FakeApp(make_points())
        app.model.converged = True
        controller = AnimationController(app)
        controller.step_once()
        self.assertEqual(app.draws, [])
        self.assertEqual(app.model.iteration, 0)

    def test_step_once_blocked_while_placing_centroids(self) -> None:
        app = FakeApp(make_points())
        app.placing_centroids = True
        controller = AnimationController(app)
        controller.step_once()
        self.assertEqual(app.draws, [])
        self.assertEqual(app.model.iteration, 0)

    def test_stop_cancels_pending_timer(self) -> None:
        app = FakeApp(make_points())
        controller = AnimationController(app, frames=8)
        controller.step_once()
        pending = controller.after_id
        self.assertIsNotNone(pending)

        controller.stop()
        self.assertFalse(controller.animating)
        self.assertFalse(controller.running)
        self.assertEqual(app.root.queue, [])
        self.assertIn(pending, app.root.cancelled)

    def test_run_until_converged_and_stop(self) -> None:
        app = FakeApp(make_points(), k=2)
        controller = AnimationController(app, frames=2, frame_ms=1, run_delay_ms=1)
        controller.toggle_run()
        self.assertTrue(controller.running)

        app.root.run_all()
        self.assertTrue(app.model.converged)
        self.assertFalse(controller.running)  # 收敛后自动停
        self.assertIn("收敛", app.status_var.get())

    def test_toggle_run_on_converged_model_does_nothing(self) -> None:
        app = FakeApp(make_points())
        app.model.converged = True
        controller = AnimationController(app)
        controller.toggle_run()
        self.assertFalse(controller.running)
        self.assertIn("收敛", app.status_var.get())
        self.assertEqual(app.root.queue, [])


if __name__ == "__main__":
    unittest.main()
