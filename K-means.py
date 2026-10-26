"""交互式二维 K-means 聚类实验室。

运行方式：python shiyank.py
自检方式：python shiyank.py --self-test
依赖：NumPy、Matplotlib（图形界面使用 Tkinter）
"""
from __future__ import annotations

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import argparse
import csv
import math
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

MIN_K = 2
MAX_K = 8
DEFAULT_COUNT = 180
MAX_ITERATIONS = 100
TOLERANCE = 1e-3
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
Point = tuple[float, float]

DATASETS = {
    "三簇高斯分布": "blobs",
    "四簇高斯分布": "four_blobs",
    "同心圆（非凸示例）": "rings",
    "笑脸轮廓（非凸示例）": "smiley",
}

DATASET_NOTES = {
    "blobs": "三组近似球形数据，适合观察 K-means 的典型效果。",
    "four_blobs": "四组高斯数据；试着改变 K，观察欠聚类和过聚类。",
    "rings": "同心圆不是凸形簇，K-means 会按直线边界切分而非识别圆环。",
    "smiley": "笑脸包含弯曲结构，适合展示 K-means 对非凸簇的局限。",
}


def generate_points(
    dataset: str,
    count: int = DEFAULT_COUNT,
    seed: int = 42,
    spread: float = 0.65,
) -> list[Point]:
    """按指定形状生成可复现的二维数据。"""
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
            points.append((5.0 + radius * math.cos(angle), 5.0 + radius * math.sin(angle)))
    elif dataset == "smiley":
        outline_count = count * 3 // 10
        eye_count = count * 3 // 10
        mouth_count = count - outline_count - eye_count
        points = []

        for _ in range(outline_count):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.gauss(4.0, spread * 0.12)
            points.append((5.0 + radius * math.cos(angle), 5.0 + radius * math.sin(angle)))

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


def assign_clusters(points: list[Point], centroids: list[Point]) -> list[int]:
    """将每个样本分配给欧氏距离最近的质心。"""
    return [
        min(
            range(len(centroids)),
            key=lambda cluster: (
                (point[0] - centroids[cluster][0]) ** 2
                + (point[1] - centroids[cluster][1]) ** 2
            ),
        )
        for point in points
    ]


def _confidence_ellipse(points: list[Point]) -> list[Point]:
    """返回包含二维正态分布约 95% 数据的协方差椭圆轮廓。"""
    if len(points) < 2:
        return []

    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    covariance_xx = sum((point[0] - mean_x) ** 2 for point in points) / (len(points) - 1)
    covariance_yy = sum((point[1] - mean_y) ** 2 for point in points) / (len(points) - 1)
    covariance_xy = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / (len(points) - 1)

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


def silhouette_score(points: list[Point], labels: list[int]) -> float | None:
    """计算平均轮廓系数；无有效多簇划分时返回 None。"""
    clusters = sorted(set(labels))
    if len(clusters) < 2 or len(clusters) >= len(points):
        return None

    members = {
        cluster: [index for index, label in enumerate(labels) if label == cluster]
        for cluster in clusters
    }
    scores = []
    for index, label in enumerate(labels):
        own_members = members[label]
        own_count = len(own_members)
        if own_count <= 1:
            scores.append(0.0)
            continue
        within = sum(
            math.dist(points[index], points[member])
            for member in own_members
            if member != index
        ) / (own_count - 1)
        nearest_other = min(
            sum(math.dist(points[index], points[member]) for member in other_members)
            / len(other_members)
            for cluster, other_members in members.items()
            if cluster != label
        )
        denominator = max(within, nearest_other)
        scores.append(
            0.0 if denominator <= 0.0 else (nearest_other - within) / denominator
        )
    return sum(scores) / len(scores)


def recommend_k_values(
    results: list[tuple[int, float, float | None]],
) -> tuple[int, int | None, float | None]:
    """根据 SSE 曲线最大弦距和轮廓系数峰值推荐 K。"""
    k_values = [result[0] for result in results]
    inertias = [result[1] for result in results]
    inertia_range = max(inertias) - min(inertias)
    if len(results) < 3 or inertia_range <= 0.0:
        elbow_k = k_values[0]
    else:
        x_range = k_values[-1] - k_values[0]
        normalized_x = [(k - k_values[0]) / x_range for k in k_values]
        normalized_sse = [
            (inertia - min(inertias)) / inertia_range for inertia in inertias
        ]
        distances = [
            abs(y - (1.0 - x)) / math.sqrt(2.0)
            for x, y in zip(normalized_x, normalized_sse)
        ]
        elbow_k = k_values[max(range(len(distances)), key=distances.__getitem__)]

    valid_silhouettes = [
        (result[0], result[2]) for result in results if result[2] is not None
    ]
    if not valid_silhouettes:
        return elbow_k, None, None
    silhouette_k, silhouette_value = max(
        valid_silhouettes, key=lambda result: result[1]
    )
    return elbow_k, silhouette_k, silhouette_value


class KMeans2D:
    """K-means 状态机：初始化、单轮更新、收敛判断与指标。"""

    def __init__(
        self,
        points: list[Point],
        k: int,
        seed: int = 42,
        initialization: str = "kmeans++",
        tolerance: float = TOLERANCE,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        try:
            self.points = [(float(point[0]), float(point[1])) for point in points]
            if any(len(point) != 2 for point in points):
                raise ValueError
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("输入数据必须是形状为 (样本数, 2) 的二维数组。") from error
        if not self.points or any(
            not math.isfinite(value) for point in self.points for value in point
        ):
            raise ValueError("输入数据必须全部为有限数值。")
        if not MIN_K <= k <= min(MAX_K, len(self.points)):
            raise ValueError(f"K 必须介于 {MIN_K} 和 {min(MAX_K, len(self.points))} 之间。")
        if initialization not in ("kmeans++", "random"):
            raise ValueError("初始化方式必须为 kmeans++ 或 random。")

        self.k = k
        self.rng = random.Random(seed)
        self.initialization = initialization
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration = 0
        self.converged = False
        self.history: list[float] = []
        self.centroids = self._initialize()
        self.labels = assign_clusters(self.points, self.centroids)
        self.inertia = self._calculate_inertia()
        self.history.append(self.inertia)

    def _initialize(self) -> list[Point]:
        """从互不相同的样本行中选取初始质心。"""
        count = len(self.points)
        if self.initialization == "random":
            indices = self.rng.sample(range(count), self.k)
        else:
            indices = [self.rng.randrange(count)]
            while len(indices) < self.k:
                distances_squared = [
                    min(
                        (point[0] - self.points[index][0]) ** 2
                        + (point[1] - self.points[index][1]) ** 2
                        for index in indices
                    )
                    for point in self.points
                ]
                for index in indices:
                    distances_squared[index] = 0.0
                total = sum(distances_squared)
                remaining = [index for index in range(count) if index not in indices]
                if total <= 0.0:
                    next_index = self.rng.choice(remaining)
                else:
                    threshold = self.rng.random() * total
                    cumulative = 0.0
                    next_index = count - 1
                    for index, distance_squared in enumerate(distances_squared):
                        cumulative += distance_squared
                        if cumulative > threshold:
                            next_index = index
                            break
                    if next_index >= count or next_index in indices:
                        next_index = remaining[-1]
                indices.append(next_index)
        return [self.points[index] for index in indices]

    def _calculate_inertia(self) -> float:
        return sum(
            (point[0] - self.centroids[label][0]) ** 2
            + (point[1] - self.centroids[label][1]) ** 2
            for point, label in zip(self.points, self.labels)
        )

    def step(self) -> tuple[list[Point], list[Point], bool]:
        """执行一轮分配与更新，返回旧质心、新质心和收敛状态。"""
        if self.converged or self.iteration >= self.max_iterations:
            return self.centroids.copy(), self.centroids.copy(), True

        old_centroids = self.centroids.copy()
        old_labels = assign_clusters(self.points, old_centroids)
        new_centroids = old_centroids.copy()
        distances_squared = [
            (point[0] - old_centroids[label][0]) ** 2
            + (point[1] - old_centroids[label][1]) ** 2
            for point, label in zip(self.points, old_labels)
        ]
        empty_clusters: list[int] = []

        for cluster in range(self.k):
            members = [
                point for point, label in zip(self.points, old_labels) if label == cluster
            ]
            if len(members):
                new_centroids[cluster] = (
                    sum(point[0] for point in members) / len(members),
                    sum(point[1] for point in members) / len(members),
                )
            else:
                empty_clusters.append(cluster)

        if empty_clusters:
            farthest_first = sorted(
                range(len(distances_squared)),
                key=distances_squared.__getitem__,
                reverse=True,
            )
            for cluster, point_index in zip(empty_clusters, farthest_first):
                new_centroids[cluster] = self.points[point_index]

        movement = [
            math.dist(new, old) for old, new in zip(old_centroids, new_centroids)
        ]
        self.centroids = new_centroids
        self.labels = assign_clusters(self.points, self.centroids)
        self.inertia = self._calculate_inertia()
        self.iteration += 1
        self.converged = (
            all(distance <= self.tolerance for distance in movement)
            or self.iteration >= self.max_iterations
        )
        self.history.append(self.inertia)
        return old_centroids, new_centroids.copy(), self.converged


class KMeansApp:
    """Tkinter 桌面界面与标准库 Canvas 二维可视化。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("K-Means Lab | 二维聚类可视化实验室")
        self.root.geometry("1240x790")
        self.root.minsize(1040, 680)
        self.root.configure(bg="#f1f5f9")

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#f1f5f9",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 19, "bold"),
        )
        style.configure(
            "Sub.TLabel",
            background="#f1f5f9",
            foreground="#64748b",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#ffffff",
            foreground="#334155",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Metric.TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        style.configure(
            "TButton",
            font=("Microsoft YaHei UI", 9),
            padding=(8, 6),
        )

        self.dataset_var = tk.StringVar(value="三簇高斯分布")
        self.k_var = tk.StringVar(value="3")
        self.init_var = tk.StringVar(value="K-means++")
        self.count_var = tk.StringVar(value=str(DEFAULT_COUNT))
        self.seed_var = tk.StringVar(value="42")
        self.spread_var = tk.DoubleVar(value=0.65)
        self.status_var = tk.StringVar(value="就绪：点击「单步迭代」观察算法更新。")
        self.metric_vars = {
            "iteration": tk.StringVar(value="0"),
            "inertia": tk.StringVar(value="—"),
            "silhouette": tk.StringVar(value="—"),
            "clusters": tk.StringVar(value="—"),
        }

        self.points = generate_points("blobs", DEFAULT_COUNT, 42, 0.65)
        self.active_dataset = "blobs"
        self.active_dataset_label = "三簇高斯分布"
        self.model = self._new_model()
        self.initial_inertia = self.model.inertia
        self.running = False
        self.after_id: str | None = None
        self.animating = False
        self.placing_centroids = False
        self.placed_centroids: list[list[float]] = []
        self._build_layout()
        self._draw()

    def _new_model(self) -> KMeans2D:
        seed = self._read_seed()
        initialization = "kmeans++" if self.init_var.get() == "K-means++" else "random"
        return KMeans2D(
            self.points,
            int(self.k_var.get()),
            seed=seed,
            initialization=initialization,
        )

    def _read_seed(self) -> int:
        try:
            seed = int(self.seed_var.get())
        except ValueError as error:
            raise ValueError("随机种子必须是整数。") from error
        if seed < 0:
            raise ValueError("随机种子必须是非负整数。")
        return seed

    def _update_sidebar_scroll_region(self, _event: object) -> None:
        if _event.widget is self.sidebar:
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _resize_sidebar(self, event: object) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _scroll_sidebar(self, event: object) -> None:
        self.sidebar_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(20, 14, 20, 8))
        header.pack(fill="x")
        ttk.Label(header, text="K-MEANS LAB", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="二维聚类 · 算法教学 · 结果分析",
            style="Sub.TLabel",
        ).pack(side="left", padx=(14, 0), pady=(6, 0))

        body = ttk.Frame(self.root, padding=(16, 6, 16, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar_container = ttk.Frame(body, width=300)
        sidebar_container.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar_container.grid_propagate(False)
        self.sidebar_canvas = tk.Canvas(
            sidebar_container,
            width=285,
            highlightthickness=0,
            bg="#ffffff",
        )
        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_container, orient="vertical", command=self.sidebar_canvas.yview
        )
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scrollbar.pack(side="right", fill="y")
        self.sidebar = ttk.Frame(
            self.sidebar_canvas, style="Card.TFrame", padding=16, width=285
        )
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar, anchor="nw"
        )
        self.sidebar.bind("<Configure>", self._update_sidebar_scroll_region)
        self.sidebar_canvas.bind("<Configure>", self._resize_sidebar)
        self.sidebar_canvas.bind("<MouseWheel>", self._scroll_sidebar)
        self._build_controls()

        workspace = ttk.Frame(body)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)

        metrics = ttk.Frame(workspace)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            metrics.columnconfigure(column, weight=1)
        self._metric_card(metrics, 0, "迭代轮数", "iteration", "#2563eb")
        self._metric_card(metrics, 1, "簇内平方和 (SSE)", "inertia", "#7c3aed")
        self._metric_card(metrics, 2, "轮廓系数", "silhouette", "#059669")
        self._metric_card(metrics, 3, "有效簇数", "clusters", "#ea580c")

        plot_card = ttk.Frame(workspace, style="Card.TFrame", padding=10)
        plot_card.grid(row=1, column=0, sticky="nsew")
        plot_card.rowconfigure(0, weight=1)
        plot_card.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor="#ffffff")
        self.axes = self.figure.add_subplot(111)
        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=plot_card)
        self.canvas = self.figure_canvas.get_tk_widget()
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.figure_canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.configure(cursor="crosshair")

        footer = ttk.Frame(workspace, style="Card.TFrame", padding=(12, 8))
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            textvariable=self.status_var,
            background="#ffffff",
            foreground="#334155",
            font=("Microsoft YaHei UI", 9),
            wraplength=780,
        ).grid(row=0, column=0, sticky="w")
        self.detail_var = tk.StringVar()
        ttk.Label(
            footer,
            textvariable=self.detail_var,
            background="#ffffff",
            foreground="#64748b",
            font=("Microsoft YaHei UI", 9),
            wraplength=780,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind(
            "<space>",
            lambda event: self._toggle_run() if event.keysym == "space" else None,
        )
        self.root.bind(
            "<Right>", lambda event: self._step() if event.keysym == "Right" else None
        )
        self.root.bind(
            "<Escape>",
            lambda event: self._cancel_placement() if event.keysym == "Escape" else None,
        )

    def _metric_card(
        self, parent: ttk.Frame, column: int, title: str, key: str, accent: str
    ) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 9))
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
        tk.Frame(card, bg=accent, height=3).pack(fill="x", pady=(0, 7))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=self.metric_vars[key], style="Metric.TLabel").pack(
            anchor="w", pady=(3, 0)
        )

    def _build_controls(self) -> None:
        ttk.Label(
            self.sidebar,
            text="实验设置",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self._field_label("数据形状")
        self.dataset_box = ttk.Combobox(
            self.sidebar,
            textvariable=self.dataset_var,
            values=list(DATASETS),
            state="readonly",
        )
        self.dataset_box.pack(fill="x", pady=(0, 8))

        self._field_label("簇数 K（2–8）")
        self.k_box = ttk.Combobox(
            self.sidebar,
            textvariable=self.k_var,
            values=[str(value) for value in range(MIN_K, MAX_K + 1)],
            state="readonly",
        )
        self.k_box.pack(fill="x", pady=(0, 8))

        self._field_label("初始质心")
        self.init_box = ttk.Combobox(
            self.sidebar,
            textvariable=self.init_var,
            values=["K-means++", "随机"],
            state="readonly",
        )
        self.init_box.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(self.sidebar, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="样本数", background="#ffffff").pack(side="left")
        ttk.Spinbox(
            row, textvariable=self.count_var, from_=60, to=500, increment=20, width=8
        ).pack(side="right")

        row = ttk.Frame(self.sidebar, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 3))
        ttk.Label(row, text="随机种子", background="#ffffff").pack(side="left")
        ttk.Spinbox(
            row, textvariable=self.seed_var, from_=0, to=999999, width=10
        ).pack(side="right")

        self._field_label("数据离散程度")
        ttk.Scale(
            self.sidebar,
            from_=0.25,
            to=1.3,
            variable=self.spread_var,
            orient="horizontal",
        ).pack(fill="x", pady=(0, 2))
        self.spread_label = ttk.Label(
            self.sidebar, text="0.65", background="#ffffff", foreground="#64748b"
        )
        self.spread_label.pack(anchor="e", pady=(0, 8))
        self.spread_var.trace_add("write", self._update_spread_label)

        ttk.Button(
            self.sidebar, text="生成数据 / 应用设置", command=self._apply_settings
        ).pack(fill="x", pady=(2, 5))

        ttk.Separator(self.sidebar).pack(fill="x", pady=10)
        ttk.Label(
            self.sidebar,
            text="运行算法",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 7))

        ttk.Button(
            self.sidebar, text="▶  连续运行 / 暂停  (空格)", command=self._toggle_run
        ).pack(fill="x", pady=3)
        ttk.Button(
            self.sidebar, text="单步迭代  (→)", command=self._step
        ).pack(fill="x", pady=3)
        ttk.Button(
            self.sidebar, text="重新初始化质心", command=self._reset_model
        ).pack(fill="x", pady=3)
        self.place_button = ttk.Button(
            self.sidebar, text="在图上手动放置质心", command=self._begin_placement
        )
        self.place_button.pack(fill="x", pady=3)

        ttk.Separator(self.sidebar).pack(fill="x", pady=10)
        ttk.Label(
            self.sidebar,
            text="分析工具",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 7))
        ttk.Button(
            self.sidebar,
            text="K 值分析（肘部＋轮廓）",
            command=self._analyze_k_values,
        ).pack(fill="x", pady=3)
        ttk.Button(
            self.sidebar, text="导出 CSV", command=self._export_csv
        ).pack(fill="x", pady=3)

        ttk.Separator(self.sidebar).pack(fill="x", pady=10)
        ttk.Label(
            self.sidebar,
            text="结果与学习提示",
            background="#ffffff",
            foreground="#0f172a",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        self.cluster_text = tk.Text(
            self.sidebar,
            height=7,
            wrap="word",
            relief="flat",
            bg="#f8fafc",
            fg="#334155",
            font=("Microsoft YaHei UI", 9),
            padx=8,
            pady=6,
            state="disabled",
        )
        self.cluster_text.pack(fill="x", pady=(0, 7))
        ttk.Label(
            self.sidebar,
            text=(
                "分配：最近质心；更新：簇内均值\n"
                "停止条件：质心移动量 ≤ 0.001 或迭代达到100轮\n"
                "彩色虚线表示簇的95%置信椭圆。\n"
                "轮廓系数范围 [-1,1]，数值越大，簇分离效果越好。"
            ),
            background="#ffffff",
            foreground="#64748b",
            font=("Microsoft YaHei UI", 8),
            justify="left",
            wraplength=250,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(
            self.sidebar, text="保存图像", command=self._save_image
        ).pack(fill="x", pady=(0, 2))

    def _field_label(self, text: str) -> None:
        ttk.Label(
            self.sidebar,
            text=text,
            background="#ffffff",
            foreground="#475569",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", pady=(2, 4))

    def _update_spread_label(self, *_trace_info: object) -> None:
        if len(_trace_info) == 3:
            self.spread_label.configure(text=f"{self.spread_var.get():.2f}")

    def _apply_settings(self) -> None:
        try:
            count = int(self.count_var.get())
            if not 60 <= count <= 500:
                raise ValueError("样本数必须在 60 到 500 之间。")
            seed = self._read_seed()
            dataset = DATASETS[self.dataset_var.get()]
            points = generate_points(
                dataset, count=count, seed=seed, spread=self.spread_var.get()
            )
            k = int(self.k_var.get())
            initialization = "kmeans++" if self.init_var.get() == "K-means++" else "random"
            model = KMeans2D(points, k, seed, initialization)
        except (KeyError, ValueError) as error:
            messagebox.showerror("设置无效", str(error), parent=self.root)
            return

        self._stop_run()
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
        self._stop_run()
        self._cancel_placement()
        self.model = model
        self.initial_inertia = model.inertia
        self.status_var.set("质心已重新初始化，迭代计数已清零。")
        self._draw()

    def _begin_placement(self) -> None:
        self._stop_run()
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
            event.inaxes is not self.axes
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
            except ValueError as error:
                messagebox.showerror("无法设置质心", str(error), parent=self.root)
                self._cancel_placement()
                return
            model.centroids = self.placed_centroids.copy()
            model.labels = assign_clusters(self.points, model.centroids)
            model.inertia = model._calculate_inertia()
            model.history = [model.inertia]
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

    def _step(self) -> None:
        if self.placing_centroids or self.animating or self.model.converged:
            return
        old_centroids, new_centroids, _ = self.model.step()
        self.animating = True
        self._animate_centroids(old_centroids, new_centroids, 0)

    def _animate_centroids(
        self, start: list[Point], target: list[Point], frame: int
    ) -> None:
        total_frames = 8
        progress = min((frame + 1) / total_frames, 1.0)
        eased = progress * progress * (3.0 - 2.0 * progress)
        current = [
            (
                before[0] + (after[0] - before[0]) * eased,
                before[1] + (after[1] - before[1]) * eased,
            )
            for before, after in zip(start, target)
        ]
        self._draw(current, movement=(start, target), refresh_metrics=False)
        if frame + 1 < total_frames:
            self.after_id = self.root.after(
                24, lambda: self._animate_centroids(start, target, frame + 1)
            )
            return

        self.after_id = None
        self.animating = False
        self._draw()
        if self.model.converged:
            status = (
                f"算法已停止：第 {self.model.iteration} 轮达到收敛条件。"
                if self.model.iteration < self.model.max_iterations
                else f"已达到最大迭代次数 {self.model.max_iterations}。"
            )
            self.status_var.set(status)
            self.running = False
        elif self.running:
            self.after_id = self.root.after(220, self._step)

    def _toggle_run(self) -> None:
        if self.placing_centroids:
            return
        if self.running:
            self.running = False
            self.status_var.set("已暂停；可以单步继续或重新开始连续运行。")
            return
        if self.model.converged:
            self.status_var.set("算法已收敛；重新初始化质心后可以再次运行。")
            return
        self.running = True
        self.status_var.set("连续运行中……")
        if not self.animating:
            self._step()

    def _stop_run(self) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.animating = False

    def _draw(
        self,
        display_centroids: list[Point] | None = None,
        movement: tuple[list[Point], list[Point]] | None = None,
        refresh_metrics: bool = True,
    ) -> None:
        centroids = (
            self.model.centroids
            if display_centroids is None
            else display_centroids
        )
        self.axes.clear()
        self.figure.subplots_adjust(right=0.78)
        self.axes.set_facecolor("#fbfdff")
        self.axes.update_datalim(((0, 0), (10, 10)))
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 10)
        self.axes.set_xticks(range(0, 11, 2))
        self.axes.set_yticks(range(0, 11, 2))
        self.axes.grid(True, alpha=0.2)
        self.axes.set_axisbelow(True)
        self.axes.set_xlabel("特征X_1", fontname="Microsoft YaHei UI")
        self.axes.set_ylabel(
            "特征X_2",
            fontname="Microsoft YaHei UI",
            rotation=0,
            labelpad=28,
        )
        self.axes.set_title(
            f"{self.active_dataset_label}  |  K={self.model.k}  |  "
            f"第 {self.model.iteration} 轮"
            + (" · 点击图表放置质心" if self.placing_centroids else ""),
            fontname="Microsoft YaHei UI",
        )

        points = np.asarray(self.points, dtype=float)
        labels = np.asarray(self.model.labels, dtype=int)
        counts = [
            int(np.count_nonzero(labels == cluster))
            for cluster in range(self.model.k)
        ]
        for cluster, color in enumerate(PALETTE[: self.model.k]):
            members_array = points[labels == cluster]
            members = [tuple(point) for point in members_array]
            ellipse = _confidence_ellipse(members)
            if ellipse:
                self.axes.plot(
                    [point[0] for point in ellipse],
                    [point[1] for point in ellipse],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )
            self.axes.scatter(
                members_array[:, 0] if len(members_array) else [],
                members_array[:, 1] if len(members_array) else [],
                s=32,
                color=color,
                label=f"簇{cluster + 1}(n={counts[cluster]})",
                zorder=2,
            )

        if movement is not None:
            start, target = movement
            for before, after in zip(start, target):
                self.axes.annotate(
                    "",
                    xy=after,
                    xytext=before,
                    arrowprops={"arrowstyle": "->", "color": "#0f172a", "lw": 1.2},
                    zorder=3,
                )

        self.axes.scatter(
            [center[0] for center in centroids],
            [center[1] for center in centroids],
            marker="x",
            c="black",
            s=150,
            linewidths=2.5,
            zorder=4,
        )
        if self.placing_centroids and self.placed_centroids:
            placed = np.asarray(self.placed_centroids, dtype=float)
            self.axes.scatter(
                placed[:, 0],
                placed[:, 1],
                marker="o",
                s=100,
                c="#f59e0b",
                edgecolors="#0f172a",
                linewidths=1.5,
                zorder=5,
            )
            for index, point in enumerate(self.placed_centroids):
                self.axes.annotate(
                    str(index + 1),
                    point,
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    zorder=6,
                )

        self.axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            framealpha=0.9,
            prop={"family": "Microsoft YaHei UI", "size": 8},
        )
        self.axes.margins(x=0.08, y=0.08)
        self.figure_canvas.draw_idle()

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
        self.metric_vars["clusters"].set(
            f"{len(actual_clusters)} / {self.model.k}"
        )

        cluster_rows = []
        for cluster in range(self.model.k):
            member_count = labels.count(cluster)
            x, y = self.model.centroids[cluster]
            cluster_rows.append(
                f"簇{cluster + 1}：{member_count}点，"
                f"质心 ({x:.2f}, {y:.2f})"
            )
        self.cluster_text.configure(state="normal")
        self.cluster_text.delete("1.0", "end")
        self.cluster_text.insert("1.0", "\n".join(cluster_rows))
        self.cluster_text.configure(state="disabled")

        reduction = 0.0
        if self.initial_inertia > 0.0:
            reduction = max(
                0.0, (self.initial_inertia - self.model.inertia) / self.initial_inertia
            )
        self.detail_var.set(
            f"{DATASET_NOTES[self.active_dataset]}  "
            f"SSE 相对初始状态下降 {reduction:.1%}。"
        )

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出聚类结果",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialfile="kmeans_results.csv",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(["x", "y", "cluster", "centroid_x", "centroid_y"])
                for point, label in zip(self.points, self.model.labels):
                    center = self.model.centroids[label]
                    writer.writerow(
                        [
                            f"{point[0]:.6f}",
                            f"{point[1]:.6f}",
                            int(label) + 1,
                            f"{center[0]:.6f}",
                            f"{center[1]:.6f}",
                        ]
                    )
        except OSError as error:
            messagebox.showerror("导出失败", str(error), parent=self.root)
            return
        self.status_var.set(f"聚类结果已导出：{path}")

    def _analyze_k_values(self) -> None:
        try:
            seed = self._read_seed()
            initialization = "kmeans++" if self.init_var.get() == "K-means++" else "random"
            results: list[tuple[int, float, float | None]] = []
            for k in range(MIN_K, MAX_K + 1):
                model = KMeans2D(
                    self.points,
                    k,
                    seed=seed,
                    initialization=initialization,
                )
                while not model.converged:
                    model.step()
                results.append(
                    (k, model.inertia, silhouette_score(self.points, model.labels))
                )
        except ValueError as error:
            messagebox.showerror("K 值分析失败", str(error), parent=self.root)
            return

        window = tk.Toplevel(self.root)
        window.title("K 值分析（肘部＋轮廓）")
        window.geometry("900x440")
        window.minsize(700, 360)
        window.transient(self.root)

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

    def _save_image(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存聚类可视化",
            defaultextension=".png",
            filetypes=[
                ("PNG 图像", "*.png"),
                ("PDF 文件", "*.pdf"),
                ("SVG 图像", "*.svg"),
                ("EPS 文件", "*.eps"),
            ],
            initialfile="kmeans_visualization.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except (OSError, ValueError) as error:
            messagebox.showerror("保存失败", str(error), parent=self.root)
            return
        self.status_var.set(f"可视化图像已保存：{path}")

    def _close(self) -> None:
        self._stop_run()
        self.root.destroy()


def run_self_tests() -> int:
    """运行无需打开窗口的算法与数据生成自检。"""
    import unittest

    class KMeansTests(unittest.TestCase):
        def test_confidence_ellipse_is_finite_and_closed(self) -> None:
            points: list[Point] = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
            ellipse = _confidence_ellipse(points)
            self.assertEqual(len(ellipse), 65)
            self.assertEqual(ellipse[0], ellipse[-1])
            self.assertTrue(
                all(math.isfinite(value) for point in ellipse for value in point)
            )
            self.assertEqual(_confidence_ellipse(points[:1]), [])

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

        def test_kmeans_converges_and_centers_are_means(self) -> None:
            first_rng = random.Random(2)
            second_rng = random.Random(3)
            points = [
                (first_rng.gauss(2.0, 0.2), first_rng.gauss(2.0, 0.2))
                for _ in range(50)
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

        def test_silhouette_and_invalid_k(self) -> None:
            points: list[Point] = [(0.0, 0.0), (0.2, 0.0), (5.0, 5.0), (5.2, 5.0)]
            score = silhouette_score(points, [0, 0, 1, 1])
            self.assertIsNotNone(score)
            self.assertGreater(score, 0.9)
            with self.assertRaises(ValueError):
                KMeans2D(points, 1)

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

        def test_duplicate_points_and_empty_clusters_are_safe(self) -> None:
            points: list[Point] = [(1.0, 2.0)] * 12
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
            self.assertIsNone(silhouette_score(points, model.labels))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(KMeansTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="交互式二维 K-means 聚类演示")
    parser.add_argument(
        "--self-test", action="store_true", help="运行算法自检而不启动图形界面"
    )
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()

    root = tk.Tk()
    KMeansApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())