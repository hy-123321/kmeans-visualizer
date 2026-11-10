"""动画控制：单步迭代的补间、连续运行/暂停、节奏与取消。

维护人：叶世航（动画与文档）

控制器约定的 app 接口：
    app.model               当前 KMeans2D 实例
    app.placing_centroids   手动放置质心时暂停动画
    app._draw(display_centroids=None, movement=None, refresh_metrics=True)
    app.status_var          状态栏文案

注意：所有 `after` 定时器都必须能被打断，否则关闭窗口时 Tk 会报错。
"""
from __future__ import annotations

from .config import (
    ANIMATION_FRAME_MS,
    ANIMATION_FRAMES,
    ANIMATION_RUN_DELAY_MS,
    Point,
)


class AnimationController:
    """负责“什么时候走一步、走一步画几帧”。"""

    def __init__(
        self,
        app,
        frames: int = ANIMATION_FRAMES,
        frame_ms: int = ANIMATION_FRAME_MS,
        run_delay_ms: int = ANIMATION_RUN_DELAY_MS,
    ) -> None:
        self.app = app
        self.frames = frames
        self.frame_ms = frame_ms
        self.run_delay_ms = run_delay_ms
        self.running = False
        self.animating = False
        self.after_id: str | None = None

    # ---- 对外按钮接口 ----------------------------------------------------
    def step_once(self) -> None:
        """单步：算法走一轮，质心平滑移动到新位置。"""
        app = self.app
        if app.placing_centroids or self.animating or app.model.converged:
            return
        old_centroids, new_centroids, _ = app.model.step()
        self.animating = True
        self._animate(old_centroids, new_centroids, 0)

    def toggle_run(self) -> None:
        """连续运行 / 暂停。"""
        app = self.app
        if app.placing_centroids:
            return
        if self.running:
            self.running = False
            app.status_var.set("已暂停；可以单步继续或重新开始连续运行。")
            return
        if app.model.converged:
            app.status_var.set("算法已收敛；重新初始化质心后可以再次运行。")
            return
        self.running = True
        app.status_var.set("连续运行中……")
        if not self.animating:
            self.step_once()

    def stop(self) -> None:
        """停止连续运行并取消待执行的定时器（切换设置、关闭窗口时调用）。"""
        self.running = False
        if self.after_id is not None:
            self.app.root.after_cancel(self.after_id)
            self.after_id = None
        self.animating = False

    # ---- 内部补间 --------------------------------------------------------
    def _animate(self, start: list[Point], target: list[Point], frame: int) -> None:
        progress = min((frame + 1) / self.frames, 1.0)
        eased = progress * progress * (3.0 - 2.0 * progress)
        current = [
            (
                before[0] + (after[0] - before[0]) * eased,
                before[1] + (after[1] - before[1]) * eased,
            )
            for before, after in zip(start, target)
        ]
        self.app._draw(current, movement=(start, target), refresh_metrics=False)
        if frame + 1 < self.frames:
            self.after_id = self.app.root.after(
                self.frame_ms, lambda: self._animate(start, target, frame + 1)
            )
            return

        self.after_id = None
        self.animating = False
        self.app._draw()

        model = self.app.model
        if model.converged:
            status = (
                f"算法已停止：第 {model.iteration} 轮达到收敛条件。"
                if model.iteration < model.max_iterations
                else f"已达到最大迭代次数 {model.max_iterations}。"
            )
            self.app.status_var.set(status)
            self.running = False
        elif self.running:
            self.after_id = self.app.root.after(self.run_delay_ms, self.step_once)
