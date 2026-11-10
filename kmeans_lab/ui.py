"""界面层：主题样式、整体布局与控件。

维护人：庞照阳（UI/UX 交互设计师）

`build_layout(app)` 负责把控件搭出来，并把按钮/快捷键绑到 app 的处理函数上。
它约定的 app 接口（控制器必须提供，详见 docs/CONTRIBUTING.md）：

    属性：root, dataset_var, k_var, init_var, count_var, seed_var,
          spread_var, status_var, detail_var, metric_vars
    方法：_apply_settings, _toggle_run, _step, _reset_model,
          _begin_placement, _analyze_k_values, _export_csv,
          _save_image, _on_plot_click, _cancel_placement
    产物：app.view（PlotView）、app.place_button、app.cluster_text
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .config import (
    BG_COLOR,
    CARD_COLOR,
    MAX_COUNT,
    MAX_K,
    MIN_COUNT,
    MIN_K,
    DATASETS,
)
from .view import PlotView

UI_FONT = "Microsoft YaHei UI"
SIDEBAR_WIDTH = 312


def configure_style(root: tk.Tk) -> None:
    """统一 ttk 主题与字号。"""
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TFrame", background=BG_COLOR)
    style.configure("Card.TFrame", background=CARD_COLOR)
    style.configure(
        "Title.TLabel",
        background=BG_COLOR,
        foreground="#0f172a",
        font=(UI_FONT, 19, "bold"),
    )
    style.configure(
        "Sub.TLabel",
        background=BG_COLOR,
        foreground="#64748b",
        font=(UI_FONT, 9),
    )
    style.configure(
        "CardTitle.TLabel",
        background=CARD_COLOR,
        foreground="#334155",
        font=(UI_FONT, 10, "bold"),
    )
    style.configure(
        "Metric.TLabel",
        background=CARD_COLOR,
        foreground="#0f172a",
        font=(UI_FONT, 15, "bold"),
    )
    style.configure("TButton", font=(UI_FONT, 9), padding=(8, 6))


def build_layout(app) -> None:
    """搭建主窗口：页头 / 左侧控制栏 / 右侧指标卡 + 画布 + 状态栏。"""
    root = app.root
    configure_style(root)

    header = ttk.Frame(root, padding=(20, 14, 20, 8))
    header.pack(fill="x")
    ttk.Label(header, text="K-MEANS LAB", style="Title.TLabel").pack(side="left")
    ttk.Label(
        header,
        text="二维聚类 · 算法教学 · 结果分析",
        style="Sub.TLabel",
    ).pack(side="left", padx=(14, 0), pady=(6, 0))

    body = ttk.Frame(root, padding=(16, 6, 16, 16))
    body.pack(fill="both", expand=True)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    _build_sidebar(app, body)
    _build_workspace(app, body)

    root.protocol("WM_DELETE_WINDOW", app._close)
    root.bind(
        "<space>",
        lambda event: app._toggle_run() if event.keysym == "space" else None,
    )
    root.bind(
        "<Right>", lambda event: app._step() if event.keysym == "Right" else None
    )
    root.bind(
        "<Escape>",
        lambda event: app._cancel_placement() if event.keysym == "Escape" else None,
    )


# --------------------------------------------------------------------------
# 左侧控制栏
# --------------------------------------------------------------------------
def _build_sidebar(app, body: ttk.Frame) -> None:
    container = ttk.Frame(body, width=SIDEBAR_WIDTH)
    container.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
    container.grid_propagate(False)

    canvas = tk.Canvas(container, width=285, highlightthickness=0, bg=CARD_COLOR)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    sidebar = ttk.Frame(canvas, style="Card.TFrame", padding=16, width=285)
    app.sidebar = sidebar
    window_id = canvas.create_window((0, 0), window=sidebar, anchor="nw")

    def _update_scroll_region(_event: object) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _resize_window(event: object) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    def _scroll(event: object) -> None:
        canvas.yview_scroll(int(-event.delta / 120), "units")

    sidebar.bind("<Configure>", _update_scroll_region)
    canvas.bind("<Configure>", _resize_window)
    canvas.bind("<MouseWheel>", _scroll)

    _build_controls(app, sidebar)


def _section_label(sidebar: ttk.Frame, text: str, size: int = 11) -> None:
    ttk.Label(
        sidebar,
        text=text,
        background=CARD_COLOR,
        foreground="#0f172a",
        font=(UI_FONT, size, "bold"),
    ).pack(anchor="w", pady=(0, 7))


def _field_label(sidebar: ttk.Frame, text: str) -> None:
    ttk.Label(
        sidebar,
        text=text,
        background=CARD_COLOR,
        foreground="#475569",
        font=(UI_FONT, 9, "bold"),
    ).pack(anchor="w", pady=(2, 4))


def _build_controls(app, sidebar: ttk.Frame) -> None:
    _section_label(sidebar, "实验设置", size=13)

    _field_label(sidebar, "数据形状")
    ttk.Combobox(
        sidebar,
        textvariable=app.dataset_var,
        values=list(DATASETS),
        state="readonly",
    ).pack(fill="x", pady=(0, 8))

    _field_label(sidebar, f"簇数 K（{MIN_K}–{MAX_K}）")
    ttk.Combobox(
        sidebar,
        textvariable=app.k_var,
        values=[str(value) for value in range(MIN_K, MAX_K + 1)],
        state="readonly",
    ).pack(fill="x", pady=(0, 8))

    _field_label(sidebar, "初始质心")
    ttk.Combobox(
        sidebar,
        textvariable=app.init_var,
        values=["K-means++", "随机"],
        state="readonly",
    ).pack(fill="x", pady=(0, 8))

    row = ttk.Frame(sidebar, style="Card.TFrame")
    row.pack(fill="x", pady=(0, 8))
    ttk.Label(row, text="样本数", background=CARD_COLOR).pack(side="left")
    ttk.Spinbox(
        row,
        textvariable=app.count_var,
        from_=MIN_COUNT,
        to=MAX_COUNT,
        increment=20,
        width=8,
    ).pack(side="right")

    row = ttk.Frame(sidebar, style="Card.TFrame")
    row.pack(fill="x", pady=(0, 3))
    ttk.Label(row, text="随机种子", background=CARD_COLOR).pack(side="left")
    ttk.Spinbox(
        row, textvariable=app.seed_var, from_=0, to=999999, width=10
    ).pack(side="right")

    _field_label(sidebar, "数据离散程度")
    ttk.Scale(
        sidebar,
        from_=0.25,
        to=1.3,
        variable=app.spread_var,
        orient="horizontal",
    ).pack(fill="x", pady=(0, 2))
    spread_label = ttk.Label(
        sidebar, text="0.65", background=CARD_COLOR, foreground="#64748b"
    )
    spread_label.pack(anchor="e", pady=(0, 8))

    def _sync_spread(*_trace_info: object) -> None:
        spread_label.configure(text=f"{app.spread_var.get():.2f}")

    app.spread_var.trace_add("write", _sync_spread)

    ttk.Button(
        sidebar, text="生成数据 / 应用设置", command=app._apply_settings
    ).pack(fill="x", pady=(2, 5))

    ttk.Separator(sidebar).pack(fill="x", pady=10)
    _section_label(sidebar, "运行算法")

    ttk.Button(
        sidebar, text="▶  连续运行 / 暂停  (空格)", command=app._toggle_run
    ).pack(fill="x", pady=3)
    ttk.Button(sidebar, text="单步迭代  (→)", command=app._step).pack(fill="x", pady=3)
    ttk.Button(
        sidebar, text="重新初始化质心", command=app._reset_model
    ).pack(fill="x", pady=3)
    app.place_button = ttk.Button(
        sidebar, text="在图上手动放置质心", command=app._begin_placement
    )
    app.place_button.pack(fill="x", pady=3)

    ttk.Separator(sidebar).pack(fill="x", pady=10)
    _section_label(sidebar, "分析工具")

    ttk.Button(
        sidebar,
        text="K 值分析（肘部＋轮廓）",
        command=app._analyze_k_values,
    ).pack(fill="x", pady=3)
    ttk.Button(sidebar, text="导出 CSV", command=app._export_csv).pack(
        fill="x", pady=3
    )

    ttk.Separator(sidebar).pack(fill="x", pady=10)
    _section_label(sidebar, "结果与学习提示", size=11)

    app.cluster_text = tk.Text(
        sidebar,
        height=7,
        wrap="word",
        relief="flat",
        bg="#f8fafc",
        fg="#334155",
        font=(UI_FONT, 9),
        padx=8,
        pady=6,
        state="disabled",
    )
    app.cluster_text.pack(fill="x", pady=(0, 7))

    ttk.Label(
        sidebar,
        text=(
            "分配：最近质心；更新：簇内均值\n"
            "停止条件：质心移动量 ≤ 0.001 或迭代达到100轮\n"
            "彩色虚线表示簇的95%置信椭圆。\n"
            "轮廓系数范围 [-1,1]，数值越大，簇分离效果越好。"
        ),
        background=CARD_COLOR,
        foreground="#64748b",
        font=(UI_FONT, 8),
        justify="left",
        wraplength=250,
    ).pack(anchor="w", pady=(0, 8))
    ttk.Button(sidebar, text="保存图像", command=app._save_image).pack(
        fill="x", pady=(0, 2)
    )


# --------------------------------------------------------------------------
# 右侧工作区
# --------------------------------------------------------------------------
def _metric_card(
    parent: ttk.Frame, column: int, title: str, variable: tk.StringVar, accent: str
) -> None:
    card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 9))
    card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
    tk.Frame(card, bg=accent, height=3).pack(fill="x", pady=(0, 7))
    ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(
        anchor="w", pady=(3, 0)
    )


def _build_workspace(app, body: ttk.Frame) -> None:
    workspace = ttk.Frame(body)
    workspace.grid(row=0, column=1, sticky="nsew")
    workspace.columnconfigure(0, weight=1)
    workspace.rowconfigure(1, weight=1)

    metrics = ttk.Frame(workspace)
    metrics.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    for column in range(4):
        metrics.columnconfigure(column, weight=1)
    _metric_card(metrics, 0, "迭代轮数", app.metric_vars["iteration"], "#2563eb")
    _metric_card(metrics, 1, "簇内平方和 (SSE)", app.metric_vars["inertia"], "#7c3aed")
    _metric_card(metrics, 2, "轮廓系数", app.metric_vars["silhouette"], "#059669")
    _metric_card(metrics, 3, "有效簇数", app.metric_vars["clusters"], "#ea580c")

    plot_card = ttk.Frame(workspace, style="Card.TFrame", padding=10)
    plot_card.grid(row=1, column=0, sticky="nsew")
    plot_card.rowconfigure(0, weight=1)
    plot_card.columnconfigure(0, weight=1)

    app.view = PlotView(plot_card)
    app.view.widget.grid(row=0, column=0, sticky="nsew")
    app.view.connect_click(app._on_plot_click)

    footer = ttk.Frame(workspace, style="Card.TFrame", padding=(12, 8))
    footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    footer.columnconfigure(0, weight=1)
    ttk.Label(
        footer,
        textvariable=app.status_var,
        background=CARD_COLOR,
        foreground="#334155",
        font=(UI_FONT, 9),
        wraplength=780,
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        footer,
        textvariable=app.detail_var,
        background=CARD_COLOR,
        foreground="#64748b",
        font=(UI_FONT, 9),
        wraplength=780,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))
