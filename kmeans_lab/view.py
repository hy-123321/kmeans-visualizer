"""可视化层：Matplotlib 画布绘制 + K 值分析弹窗。

维护人：庞照阳（UI/UX 交互设计师）

本文件只负责“把数据画出来”，不做任何算法计算、不读写文件、不控制动画节奏。
外部只需要构造一个 `Scene` 数据快照，然后 `PlotView.draw(scene)`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 中文字体与负号：在创建任何 Figure 之前设好，避免中文变成方块
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from .config import PALETTE, Point
from .metrics import recommend_k_values

FONT = "Microsoft YaHei UI"


@dataclass(frozen=True)
class Scene:
    """一次绘制的完整快照。

    字段全部是“当前这一帧要显示的数值”，绘制层不回头去读 app 或 model，
    这样动画补间时可以直接传插值后的质心坐标。
    """

    points: list[Point]
    labels: list[int]
    k: int
    centroids: list[Point]
    iteration: int
    dataset_label: str
    placing: bool = False
    placed: Sequence[Point] = field(default_factory=tuple)
    movement: tuple[list[Point], list[Point]] | None = None


def confidence_ellipse(members: list[Point]) -> list[Point]:
    """返回包含二维正态分布约 95% 数据的协方差椭圆轮廓（65 个点，首尾闭合）。"""
    if len(members) < 2:
        return []

    mean_x = sum(point[0] for point in members) / len(members)
    mean_y = sum(point[1] for point in members) / len(members)
    covariance_xx = sum((point[0] - mean_x) ** 2 for point in members) / (len(members) - 1)
    covariance_yy = sum((point[1] - mean_y) ** 2 for point in members) / (len(members) - 1)
    covariance_xy = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in members
    ) / (len(members) - 1)

    half_difference = (covariance_xx - covariance_yy) / 2.0
    radius = math.hypot(half_difference, covariance_xy)
    major_variance = max(0.0, (covariance_xx + covariance_yy) / 2.0 + radius)
    minor_variance = max(0.0, (covariance_xx + covariance_yy) / 2.0 - radius)
    angle = 0.5 * math.atan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
    major_axis = math.sqrt(5.991 * major_variance)
    minor_axis = math.sqrt(5.991 * minor_variance)
    cosine, sine = math.cos(angle), math.sin(angle)

    return [
        (
            mean_x + major_axis * math.cos(step * math.tau / 64) * cosine
            - minor_axis * math.sin(step * math.tau / 64) * sine,
            mean_y + major_axis * math.cos(step * math.tau / 64) * sine
            + minor_axis * math.sin(step * math.tau / 64) * cosine,
        )
        for step in range(65)
    ]


class PlotView:
    """主画布：负责散点、置信椭圆、质心与移动箭头的绘制。"""

    def __init__(self, master) -> None:
        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor="#ffffff")
        self.axes = self.figure.add_subplot(111)
        self.bridge = FigureCanvasTkAgg(self.figure, master=master)
        self.widget = self.bridge.get_tk_widget()
        self.widget.configure(cursor="crosshair")

    def connect_click(self, handler) -> None:
        """把鼠标点击事件交给控制器（用于手动放置质心）。"""
        self.bridge.mpl_connect("button_press_event", handler)

    def draw(self, scene: Scene) -> None:
        axes = self.axes
        axes.clear()
        self.figure.subplots_adjust(right=0.78)
        axes.set_facecolor("#fbfdff")
        axes.update_datalim(((0, 0), (10, 10)))
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlim(0, 10)
        axes.set_ylim(0, 10)
        axes.set_xticks(range(0, 11, 2))
        axes.set_yticks(range(0, 11, 2))
        axes.grid(True, alpha=0.2)
        axes.set_axisbelow(True)
        axes.set_xlabel("特征X_1", fontname=FONT)
        axes.set_ylabel("特征X_2", fontname=FONT, rotation=0, labelpad=28)
        axes.set_title(
            f"{scene.dataset_label}  |  K={scene.k}  |  第 {scene.iteration} 轮"
            + (" · 点击图表放置质心" if scene.placing else ""),
            fontname=FONT,
        )

        points = np.asarray(scene.points, dtype=float)
        labels = np.asarray(scene.labels, dtype=int)
        counts = [
            int(np.count_nonzero(labels == cluster)) for cluster in range(scene.k)
        ]
        for cluster, color in enumerate(PALETTE[: scene.k]):
            members_array = points[labels == cluster]
            ellipse = confidence_ellipse([tuple(point) for point in members_array])
            if ellipse:
                axes.plot(
                    [point[0] for point in ellipse],
                    [point[1] for point in ellipse],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )
            axes.scatter(
                members_array[:, 0] if len(members_array) else [],
                members_array[:, 1] if len(members_array) else [],
                s=32,
                color=color,
                label=f"簇{cluster + 1}(n={counts[cluster]})",
                zorder=2,
            )

        if scene.movement is not None:
            start, target = scene.movement
            for before, after in zip(start, target):
                axes.annotate(
                    "",
                    xy=after,
                    xytext=before,
                    arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.2},
                    zorder=3,
                )

        axes.scatter(
            [center[0] for center in scene.centroids],
            [center[1] for center in scene.centroids],
            marker="x",
            c="black",
            s=150,
            linewidths=2.5,
            zorder=4,
        )

        if scene.placing and scene.placed:
            placed = np.asarray(scene.placed, dtype=float)
            axes.scatter(
                placed[:, 0],
                placed[:, 1],
                marker="o",
                s=100,
                c="#f59e0b",
                edgecolors="#0f172a",
                linewidths=1.5,
                zorder=5,
            )
            for index, point in enumerate(scene.placed):
                axes.annotate(
                    str(index + 1),
                    point,
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    zorder=6,
                )

        axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            framealpha=0.9,
            prop={"family": FONT, "size": 8},
        )
        axes.margins(x=0.08, y=0.08)
        self.bridge.draw_idle()

    def savefig(self, path: str) -> None:
        self.figure.savefig(path, dpi=200, bbox_inches="tight")


def show_k_analysis(
    parent,
    results: list[tuple[int, float, float | None]],
) -> str:
    """弹出“K 值分析”窗口并返回综合建议文案。"""
    import tkinter as tk
    from tkinter import ttk

    window = tk.Toplevel(parent)
    window.title("K 值分析（肘部＋轮廓）")
    window.geometry("900x440")
    window.minsize(700, 360)
    window.transient(parent)

    figure = Figure(figsize=(9, 4.4), dpi=100, facecolor="#ffffff")
    inertia_axes, silhouette_axes = figure.subplots(1, 2)
    k_values = [result[0] for result in results]
    inertias = [result[1] for result in results]
    silhouettes = [
        float("nan") if result[2] is None else result[2] for result in results
    ]
    elbow_k, silhouette_k, silhouette_value = recommend_k_values(results)
    elbow_index = k_values.index(elbow_k)

    inertia_axes.plot(k_values, inertias, marker="o", color="#2563eb")
    inertia_axes.set_title("肘部法：簇内平方和（SSE）")
    inertia_axes.set_xlabel("簇数 K")
    inertia_axes.set_ylabel("SSE")
    inertia_axes.set_xticks(k_values)
    inertia_axes.set_xlim(k_values[0] - 0.4, k_values[-1] + 0.4)
    inertia_axes.set_ylim(bottom=0.0)
    inertia_axes.grid(True, alpha=0.25)
    inertia_axes.scatter(
        elbow_k,
        inertias[elbow_index],
        marker="*",
        s=180,
        color="#dc2626",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )
    inertia_axes.annotate(
        f"肘部 K={elbow_k}",
        xy=(elbow_k, inertias[elbow_index]),
        xytext=(8, 12),
        textcoords="offset points",
        color="#b91c1c",
        arrowprops={"arrowstyle": "->", "color": "#b91c1c"},
    )

    silhouette_axes.plot(k_values, silhouettes, marker="o", color="#059669")
    silhouette_axes.set_title("轮廓系数")
    silhouette_axes.set_xlabel("簇数 K")
    silhouette_axes.set_ylabel("平均轮廓系数")
    silhouette_axes.set_xticks(k_values)
    silhouette_axes.set_xlim(k_values[0] - 0.4, k_values[-1] + 0.4)
    silhouette_axes.set_ylim(-1.0, 1.0)
    silhouette_axes.grid(True, alpha=0.25)
    if silhouette_k is not None and silhouette_value is not None:
        silhouette_axes.scatter(
            silhouette_k,
            silhouette_value,
            marker="*",
            s=180,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        silhouette_axes.annotate(
            f"峰值 K={silhouette_k}",
            xy=(silhouette_k, silhouette_value),
            xytext=(8, 12),
            textcoords="offset points",
            color="#b91c1c",
            arrowprops={"arrowstyle": "->", "color": "#b91c1c"},
        )

    figure.tight_layout(pad=2.0)
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    canvas.draw()

    if silhouette_k is None or silhouette_value is None:
        recommendation = (
            f"肘部法推荐 K={elbow_k}（SSE={inertias[elbow_index]:.2f}）；"
            "轮廓系数没有有效结果，综合建议暂取肘部 K。"
        )
    elif silhouette_k == elbow_k:
        recommendation = (
            f"肘部法与轮廓系数一致，综合推荐 K={elbow_k}"
            f"（轮廓系数 {silhouette_value:.3f}，SSE {inertias[elbow_index]:.2f}）。"
        )
    else:
        recommendation = (
            f"综合建议 K={silhouette_k}（轮廓系数最高 {silhouette_value:.3f}）；"
            f"肘部法提示 K={elbow_k}（SSE={inertias[elbow_index]:.2f}）。"
            "两项指标不一致，建议结合业务可解释性判断。"
        )
    ttk.Label(
        window,
        text=recommendation,
        wraplength=860,
        justify="left",
        padding=(12, 4, 12, 12),
    ).pack(fill="x")
    return recommendation
