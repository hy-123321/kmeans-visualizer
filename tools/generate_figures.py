"""从真实运行结果生成文档用图。

用法（在项目根目录）：
    python tools/generate_figures.py

输出到 images/ 目录，供 README 与 docs/MANUAL.md 引用。
数据、初始质心、迭代过程都用项目自身的模块，因此图与程序运行结果完全一致，
需要更新文档配图时重跑本脚本即可，不必手工截图。
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kmeans_lab.algorithm import KMeans2D  # noqa: E402
from kmeans_lab.config import DATASET_LABELS  # noqa: E402
from kmeans_lab.data import generate_points  # noqa: E402
from kmeans_lab.metrics import silhouette_score  # noqa: E402
from kmeans_lab.view import PlotView, Scene, build_k_analysis_figure  # noqa: E402

OUT_DIR = REPO_ROOT / "images"
SEED = 42
COUNT = 180
SPREAD = 0.65

# (输出文件名, 数据集, K, 说明)
CASES = [
    ("01_blobs_k3.png", "blobs", 3),
    ("02_four_blobs_k2.png", "four_blobs", 2),
    ("03_four_blobs_k4.png", "four_blobs", 4),
    ("04_four_blobs_k6.png", "four_blobs", 6),
    ("05_rings_k2.png", "rings", 2),
    ("06_smiley_k5.png", "smiley", 5),
]


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    view = PlotView(root)

    for filename, dataset, k in CASES:
        points = generate_points(dataset, COUNT, SEED, SPREAD)
        model = KMeans2D(points, k, seed=SEED)
        model.run_to_convergence()
        view.draw(
            Scene(
                points=points,
                labels=model.labels,
                k=k,
                centroids=model.centroids,
                iteration=model.iteration,
                dataset_label=DATASET_LABELS[dataset],
            )
        )
        path = OUT_DIR / filename
        view.savefig(str(path))
        score = silhouette_score(points, model.labels)
        print(
            f"{filename:24s} K={k}  轮数={model.iteration:2d}  "
            f"SSE={model.inertia:8.2f}  轮廓={score:+.3f}"
        )

    # K 值分析双联图
    points = generate_points("four_blobs", COUNT, SEED, SPREAD)
    results = []
    for k in range(2, 9):
        model = KMeans2D(points, k, seed=SEED)
        model.run_to_convergence()
        results.append((k, model.inertia, silhouette_score(points, model.labels)))
    figure, recommendation = build_k_analysis_figure(results)
    figure.savefig(OUT_DIR / "07_k_analysis.png", dpi=200, bbox_inches="tight")
    print(f"{'07_k_analysis.png':24s} 建议：{recommendation}")

    root.destroy()
    print(f"\n已生成到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
