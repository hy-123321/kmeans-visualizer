"""K-Means 聚类可视化实验室 —— 程序入口。

为了保持 README 里的 `python K-means.py` 用法不变，这里只做一层薄包装：
真正的代码在 kmeans_lab/ 包里，按成员分工拆成了多个模块。

    python K-means.py              启动图形界面
    python K-means.py --self-test  运行单元测试
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kmeans_lab.main import main  # noqa: E402  (必须在 sys.path 设置之后导入)

if __name__ == "__main__":
    raise SystemExit(main())
