"""K-Means 聚类可视化实验室（模块化版本）。

包结构按《团队分工协作表》切分，每个模块只有一个负责人：
    config.py          袁恒宇  共享常量与接口契约
    data.py            张迎奥  数据生成
    initialization.py  张迎奥  质心初始化
    metrics.py         张迎奥  SSE / 轮廓系数 / K 值推荐
    algorithm.py       袁恒宇  K-Means 核心迭代
    view.py            庞照阳  Matplotlib 视图与绘制
    ui.py              庞照阳  Tkinter 布局与控件
    animation.py       叶世航  单步 / 连续运行的动画控制
    export.py          叶世航  CSV 与图像导出
    app.py             袁恒宇  控制器（把上面几层粘起来）
    main.py            袁恒宇  命令行入口
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
