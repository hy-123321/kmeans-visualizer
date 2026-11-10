"""文件导出：聚类结果 CSV 与可视化图像。

维护人：叶世航（动画与文档）

拆成两层，方便测试：
* `write_csv(...)`        纯函数式写文件，不弹任何对话框，单元测试直接调它；
* `export_csv_dialog(...)` 负责弹保存对话框，返回路径或 None。
"""
from __future__ import annotations

import csv
from tkinter import filedialog, messagebox

from .config import Point

CSV_HEADER = ["x", "y", "cluster", "centroid_x", "centroid_y"]


def write_csv(
    path: str,
    points: list[Point],
    labels: list[int],
    centroids: list[Point],
) -> None:
    """把每个样本的坐标、簇号（从 1 开始）和所属质心写成 csv。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(CSV_HEADER)
        for point, label in zip(points, labels):
            center = centroids[label]
            writer.writerow(
                [
                    f"{point[0]:.6f}",
                    f"{point[1]:.6f}",
                    int(label) + 1,
                    f"{center[0]:.6f}",
                    f"{center[1]:.6f}",
                ]
            )


def export_csv_dialog(
    parent,
    points: list[Point],
    labels: list[int],
    centroids: list[Point],
) -> str | None:
    """弹出保存对话框并导出 CSV；用户取消或失败时返回 None。"""
    path = filedialog.asksaveasfilename(
        parent=parent,
        title="导出聚类结果",
        defaultextension=".csv",
        filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        initialfile="kmeans_results.csv",
    )
    if not path:
        return None
    try:
        write_csv(path, points, labels, centroids)
    except OSError as error:
        messagebox.showerror("导出失败", str(error), parent=parent)
        return None
    return path


def save_image_dialog(parent, view) -> str | None:
    """弹出保存对话框并导出当前可视化图像；用户取消或失败时返回 None。"""
    path = filedialog.asksaveasfilename(
        parent=parent,
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
        return None
    try:
        view.savefig(path)
    except (OSError, ValueError) as error:
        messagebox.showerror("保存失败", str(error), parent=parent)
        return None
    return path
