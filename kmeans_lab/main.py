"""命令行入口。

维护人：袁恒宇（PM）

    python K-means.py              启动图形界面
    python K-means.py --self-test  只跑单元测试，不弹窗口
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def run_self_tests() -> int:
    """运行 tests/ 下的全部单元测试，返回进程退出码。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    suite = unittest.TestLoader().discover(str(TESTS_DIR), top_level_dir=str(REPO_ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="交互式二维 K-means 聚类演示")
    parser.add_argument(
        "--self-test", action="store_true", help="运行算法自检而不启动图形界面"
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_tests()

    import tkinter as tk

    from .app import KMeansApp

    root = tk.Tk()
    KMeansApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
