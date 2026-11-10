"""共享常量与类型别名。

维护人：袁恒宇（PM）

本文件是全项目**唯一的公共契约**。任何成员需要新增/修改常量时，
先在此文件提改动，由 PM 审核后再合并，避免多人同时改同一份配置文件。
其他模块只允许 `from .config import ...`，不允许自己另立常量。
"""
from __future__ import annotations

# ---- 算法参数 ----------------------------------------------------------
MIN_K = 2
MAX_K = 8
DEFAULT_COUNT = 180
MIN_COUNT = 60
MAX_COUNT = 500
MAX_ITERATIONS = 100
TOLERANCE = 1e-3

# ---- 界面参数 ----------------------------------------------------------
WINDOW_TITLE = "K-Means Lab | 二维聚类可视化实验室"
WINDOW_SIZE = "1240x790"
WINDOW_MIN_SIZE = (1040, 680)
BG_COLOR = "#f1f5f9"
CARD_COLOR = "#ffffff"

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

# ---- 类型别名 ----------------------------------------------------------
Point = tuple[float, float]

# ---- 数据集清单（下拉框显示名 -> 内部键） ------------------------------
DATASETS = {
    "三簇高斯分布": "blobs",
    "四簇高斯分布": "four_blobs",
    "同心圆（非凸示例）": "rings",
    "笑脸轮廓（非凸示例）": "smiley",
}
DATASET_LABELS = {value: key for key, value in DATASETS.items()}
DEFAULT_DATASET = "blobs"

DATASET_NOTES = {
    "blobs": "三组近似球形数据，适合观察 K-means 的典型效果。",
    "four_blobs": "四组高斯数据；试着改变 K，观察欠聚类和过聚类。",
    "rings": "同心圆不是凸形簇，K-means 会按直线边界切分而非识别圆环。",
    "smiley": "笑脸包含弯曲结构，适合展示 K-means 对非凸簇的局限。",
}

# ---- 动画参数（叶世航） ------------------------------------------------
ANIMATION_FRAMES = 8
ANIMATION_FRAME_MS = 24
ANIMATION_RUN_DELAY_MS = 220
