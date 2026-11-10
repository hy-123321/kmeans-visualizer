"""控制器：把界面、算法、动画、导出串起来。

维护人：袁恒宇（PM）

这一层是唯一的“胶水层”，也是合并冲突最集中的地方：
任何跨模块的改动都先在这里对齐接口，再分别通知对应负责人改自己的文件。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from matplotlib.backend_bases import MouseEvent

from .algorithm import KMeans2D, assign_clusters
from .animation import AnimationController
from .config import (
    BG_COLOR,
    DATASET_LABELS,
    DATASET_NOTES,
    DATASETS,
    DEFAULT_COUNT,
    DEFAULT_DATASET,
    MAX_COUNT,
    MAX_K,
    MIN_COUNT,
    MIN_K,
    WINDOW_MIN_SIZE,
    WINDOW_SIZE,
    WINDOW_TITLE,
    Point,
)
from .data import generate_points
from .export import export_csv_dialog, save_image_dialog
from .metrics import relative_reduction, silhouette_score
from .ui import build_layout
from .view import Scene, show_k_analysis


class KMeansApp:
    """主窗口控制器（状态 + 事件处理）。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(WINDOW_TITLE)
        root.geometry(WINDOW_SIZE)
        root.minsize(*WINDOW_MIN_SIZE)
        root.configure(bg=BG_COLOR)

        # ---- 界面状态变量（由 ui.py 读取） --------------------------------
        self.dataset_var = tk.StringVar(value=DATASET_LABELS[DEFAULT_DATASET])
        self.k_var = tk.StringVar(value="3")
        self.init_var = tk.StringVar(value="K-means++")
        self.count_var = tk.StringVar(value=str(DEFAULT_COUNT))
        self.seed_var = tk.StringVar(value="42")
        self.spread_var = tk.DoubleVar(value=0.65)
        self.status_var = tk.StringVar(value="就绪：点击「单步迭代」观察算法更新。")
        self.detail_var = tk.StringVar()
        self.metric_vars = {
            "iteration": tk.StringVar(value="0"),
            "inertia": tk.StringVar(value="—"),
            "silhouette": tk.StringVar(value="—"),
            "clusters": tk.StringVar(value="—"),
        }

        # ---- 业务状态 ----------------------------------------------------
        self.points: list[Point] = generate_points(
            DEFAULT_DATASET, DEFAULT_COUNT, 42, 0.65
        )
        self.active_dataset = DEFAULT_DATASET
        self.active_dataset_label = DATASET_LABELS[DEFAULT_DATASET]
        self.model = self._new_model()
        self.initial_inertia = self.model.inertia
        self.placing_centroids = False
        self.placed_centroids: list[Point] = []

        # ---- 视图与动画 --------------------------------------------------
        self.animator = AnimationController(self)
        build_layout(self)
        self._draw()

    # ---- 模型构造 --------------------------------------------------------
    def _initialization(self) -> str:
        return "kmeans++" if self.init_var.get() == "K-means++" else "random"

    def _new_model(self) -> KMeans2D:
        return KMeans2D(
            self.points,
            int(self.k_var.get()),
            seed=self._read_seed(),
            initialization=self._initialization(),
        )

    def _read_seed(self) -> int:
        try:
            seed = int(self.seed_var.get())
        except ValueError as error:
            raise ValueError("随机种子必须是整数。") from error
        if seed < 0:
            raise ValueError("随机种子必须是非负整数。")
        return seed

    # ---- 绘制与指标刷新 --------------------------------------------------
    def _draw(
        self,
        display_centroids: list[Point] | None = None,
        movement: tuple[list[Point], list[Point]] | None = None,
        refresh_metrics: bool = True,
    ) -> None:
        centroids = (
            self.model.centroids if display_centroids is None else display_centroids
        )
        self.view.draw(
            Scene(
                points=self.points,
                labels=self.model.labels,
                k=self.model.k,
                centroids=list(centroids),
                iteration=self.model.iteration,
                dataset_label=self.active_dataset_label,
                placing=self.placing_centroids,
                placed=tuple(self.placed_centroids),
                movement=movement,
            )
        )
        if refresh_metrics:
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        labels = self.model.labels
        actual_clusters = set(labels)
        silhouette = silhouette_score(self.points, labels)
        self.metric_vars["iteration"].set(str(self.model.iteration))
        self.metric_vars["inertia"].set(f"{self.model.inertia:.2f}")
        self.metric_vars["silhouette"].set(
            "N/A" if silhouette is None else f"{silhouette:.3f}"
        )
        self.metric_vars["clusters"].set(f"{len(actual_clusters)} / {self.model.k}")

        cluster_rows = []
        for cluster in range(self.model.k):
            member_count = labels.count(cluster)
            x, y = self.model.centroids[cluster]
            cluster_rows.append(
                f"簇{cluster + 1}：{member_count}点，质心 ({x:.2f}, {y:.2f})"
            )
        self.cluster_text.configure(state="normal")
        self.cluster_text.delete("1.0", "end")
        self.cluster_text.insert("1.0", "\n".join(cluster_rows))
        self.cluster_text.configure(state="disabled")

        reduction = relative_reduction(self.initial_inertia, self.model.inertia)
        self.detail_var.set(
            f"{DATASET_NOTES[self.active_dataset]}  "
            f"SSE 相对初始状态下降 {reduction:.1%}。"
        )

    # ---- 设置相关 --------------------------------------------------------
    def _apply_settings(self) -> None:
        try:
            count = int(self.count_var.get())
            if not MIN_COUNT <= count <= MAX_COUNT:
                raise ValueError(f"样本数必须在 {MIN_COUNT} 到 {MAX_COUNT} 之间。")
            seed = self._read_seed()
            dataset = DATASETS[self.dataset_var.get()]
            points = generate_points(
                dataset, count=count, seed=seed, spread=self.spread_var.get()
            )
            k = int(self.k_var.get())
            model = KMeans2D(points, k, seed, self._initialization())
        except (KeyError, ValueError) as error:
            messagebox.showerror("设置无效", str(error), parent=self.root)
            return

        self.animator.stop()
        self._cancel_placement()
        self.points = points
        self.active_dataset = dataset
        self.active_dataset_label = self.dataset_var.get()
        self.model = model
        self.initial_inertia = model.inertia
        self.status_var.set("已应用设置；单步执行或连续运行以开始聚类。")
        self._draw()

    def _reset_model(self) -> None:
        try:
            model = self._new_model()
        except ValueError as error:
            messagebox.showerror("无法初始化", str(error), parent=self.root)
            return
        self.animator.stop()
        self._cancel_placement()
        self.model = model
        self.initial_inertia = model.inertia
        self.status_var.set("质心已重新初始化，迭代计数已清零。")
        self._draw()

    # ---- 手动放置质心 ----------------------------------------------------
    def _begin_placement(self) -> None:
        self.animator.stop()
        self.placing_centroids = True
        self.placed_centroids = []
        self.place_button.configure(text="请在图上依次点击 K 个位置")
        self.status_var.set(
            f"手动初始化：在图表内依次点击 {self.model.k} 个位置；按 Esc 取消。"
        )
        self._draw()

    def _cancel_placement(self) -> None:
        if not self.placing_centroids:
            return
        self.placing_centroids = False
        self.placed_centroids = []
        if hasattr(self, "place_button"):
            self.place_button.configure(text="在图上手动放置质心")

    def _on_plot_click(self, event: MouseEvent) -> None:
        if not self.placing_centroids:
            return
        if (
            event.inaxes is not self.view.axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x, y = event.xdata, event.ydata
        if not 0.0 <= x <= 10.0 or not 0.0 <= y <= 10.0:
            return
        self.placed_centroids.append((x, y))
        if len(self.placed_centroids) == self.model.k:
            try:
                model = KMeans2D(
                    self.points,
                    self.model.k,
                    seed=self._read_seed(),
                    initialization="random",
                )
                model.set_centroids(self.placed_centroids.copy())
            except ValueError as error:
                messagebox.showerror("无法设置质心", str(error), parent=self.root)
                self._cancel_placement()
                return
            self.model = model
            self.initial_inertia = model.inertia
            self.placing_centroids = False
            self.place_button.configure(text="在图上手动放置质心")
            self.status_var.set("手动质心已就位；点击单步或连续运行。")
        else:
            self.status_var.set(
                f"已放置 {len(self.placed_centroids)}/{self.model.k} 个质心。"
            )
        self._draw()

    # ---- 运行控制（转交动画控制器） --------------------------------------
    def _step(self) -> None:
        self.animator.step_once()

    def _toggle_run(self) -> None:
        self.animator.toggle_run()

    # ---- 分析工具 --------------------------------------------------------
    def _analyze_k_values(self) -> None:
        try:
            seed = self._read_seed()
            initialization = self._initialization()
            results: list[tuple[int, float, float | None]] = []
            for k in range(MIN_K, MAX_K + 1):
                model = KMeans2D(
                    self.points, k, seed=seed, initialization=initialization
                )
                model.run_to_convergence()
                results.append(
                    (k, model.inertia, silhouette_score(self.points, model.labels))
                )
        except ValueError as error:
            messagebox.showerror("K 值分析失败", str(error), parent=self.root)
            return
        show_k_analysis(self.root, results)

    def _export_csv(self) -> None:
        path = export_csv_dialog(
            self.root, self.points, self.model.labels, self.model.centroids
        )
        if path:
            self.status_var.set(f"聚类结果已导出：{path}")

    def _save_image(self) -> None:
        path = save_image_dialog(self.root, self.view)
        if path:
            self.status_var.set(f"可视化图像已保存：{path}")

    # ---- 关闭 ------------------------------------------------------------
    def _close(self) -> None:
        self.animator.stop()
        self.root.destroy()


__all__ = ["KMeansApp", "assign_clusters"]
