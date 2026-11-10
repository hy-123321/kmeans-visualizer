# 协作规范与合并指南

> 维护人：袁恒宇（PM） · 适用于所有成员

## 一、代码归属表（谁只改谁的文件）

| 文件 | 负责人 | 职责 |
|---|---|---|
| `kmeans_lab/config.py` | 袁恒宇 | 常量、类型、数据集清单（公共契约，改前报备） |
| `kmeans_lab/algorithm.py` | 袁恒宇 | `KMeans2D` 状态机、`assign_clusters` |
| `kmeans_lab/app.py` | 袁恒宇 | 控制器：状态 + 事件处理 + 粘合各层 |
| `kmeans_lab/main.py` | 袁恒宇 | 命令行入口、测试调度 |
| `kmeans_lab/data.py` | 张迎奥 | 四种数据形状生成 |
| `kmeans_lab/initialization.py` | 张迎奥 | 随机 / K-means++ 初始化 |
| `kmeans_lab/metrics.py` | 张迎奥 | SSE、轮廓系数、K 值推荐 |
| `kmeans_lab/view.py` | 庞照阳 | Matplotlib 画布绘制、置信椭圆、K 值分析弹窗 |
| `kmeans_lab/ui.py` | 庞照阳 | ttk 主题、左侧控制栏、指标卡、状态栏 |
| `kmeans_lab/animation.py` | 叶世航 | 单步补间、连续运行、定时器取消 |
| `kmeans_lab/export.py` | 叶世航 | CSV 与图像导出 |
| `tests/test_data.py` `test_metrics.py` | 张迎奥 | 自己模块的测试 |
| `tests/test_algorithm.py` | 袁恒宇 | 算法测试 |
| `tests/test_view.py` | 庞照阳 | 绘制几何测试 |
| `tests/test_integration.py` | 叶世航 | 端到端联调、导出测试 |
| `docs/MANUAL.md` | 庞照阳 | 操作手册 |
| `docs/AI_Prompts_Log.md` | 叶世航 | AI 对话记录 |
| `README.md` `docs/CONTRIBUTING.md` | 袁恒宇 | 项目说明、协作规范 |

**约定：只修改自己负责的文件。** 需要调整别人的接口时先在群里同步，由本人修改；PM 只做代码合并与冲突裁决，不替成员写业务代码。

## 二、接口契约（跨文件调用只能依赖这些）

### 1. `config.py` 提供

`MIN_K / MAX_K / MIN_COUNT / MAX_COUNT / DEFAULT_COUNT / MAX_ITERATIONS / TOLERANCE / PALETTE / Point / DATASETS / DATASET_LABELS / DATASET_NOTES / DEFAULT_DATASET / ANIMATION_*`

### 2. `algorithm.KMeans2D` 提供

```python
KMeans2D(points, k, seed=42, initialization="kmeans++", tolerance=1e-3, max_iterations=100)
#属性：points / k / centroids / labels / inertia(SSE) / iteration / converged / history
#方法：step() -> (旧质心, 新质心, 是否收敛)
#      set_centroids(centroids)   手动放置质心后接管
#      run_to_convergence()       不做动画直接跑完
```

### 3. `view.PlotView` 提供

```python
PlotView(master)          # .figure / .axes / .widget / .bridge
connect_click(handler)    # 交回控制器处理点击
draw(Scene)               # Scene 见 view.py 顶部 dataclass，纯数据快照
savefig(path)
show_k_analysis(parent, results) -> str
```

### 4. `ui.build_layout(app)` 要求的 app 接口

属性：`root / dataset_var / k_var / init_var / count_var / seed_var / spread_var / status_var / detail_var / metric_vars`
方法：`_apply_settings / _toggle_run / _step / _reset_model / _begin_placement / _cancel_placement / _on_plot_click / _analyze_k_values / _export_csv / _save_image / _close`
产物：`app.view`、`app.place_button`、`app.cluster_text`

### 5. `animation.AnimationController` 要求的 app 接口

`app.model`、`app.placing_centroids`、`app.root`、`app.status_var`、`app._draw(display_centroids, movement, refresh_metrics)`

## 三、分支与提交

分支规划（每人在自己分支上开发，只推自己的文件）：

| 分支 | 负责人 |
|---|---|
| `feature/core` | 袁恒宇（算法 + 控制器） |
| `feature/ui` | 庞照阳 |
| `feature/algorithm` | 张迎奥 |
| `feature/animation` | 叶世航 |
| `docs/requirements` | 文档改动 |

首次拉分支（在 `main` 已包含本次拆分提交之后）：

```bash
git checkout main
git pull
git checkout -b feature/ui          # 换成自己的分支名
```

提交信息格式，**必须带姓名和类型**，便于在 `git log` 中追溯每处改动的负责人：

```
[庞照阳] feat(ui): 左侧控制栏改为可滚动布局
[张迎奥] feat(core): 增加 K-means++ 初始化方式
[叶世航] feat(animation): 单步迭代支持 8 帧缓动
[袁恒宇] merge: 合并 feature/ui 到 main
```

类型取值：`feat` 新功能 / `fix` 修 bug / `docs` 文档 / `test` 测试 / `refactor` 重构 / `merge` 合并。

提交前必跑：

```bash
python K-means.py --self-test     # 21 个单元测试必须全绿
python K-means.py                 # 界面能起、能单步、能连续运行
```

推送并申请合并：

```bash
git add kmeans_lab/ui.py kmeans_lab/view.py
git commit -m "[庞照阳] feat(ui): ..."
git push -u origin feature/ui
```

然后在 GitHub 上发起 Pull Request，目标分支 `main`，标题写 `[姓名] 内容`。

## 四、PM 的合并顺序（建议）

1. `feature/algorithm`（张迎奥）—— 底层，无依赖
2. `feature/ui`（庞照阳）—— 只依赖 `view.py` / `config.py`
3. `feature/animation`（叶世航）—— 依赖 app 接口
4. PM 自己的 `feature/core` 最后合并，统一改 `app.py` 适配

```bash
git checkout main
git merge --no-ff feature/algorithm
git merge --no-ff feature/ui
git merge --no-ff feature/animation
git merge --no-ff feature/core
python K-means.py --self-test
```

冲突处理原则：**谁的代码冲突，就叫谁来看**，不要 PM 单方面覆盖。`config.py` 和 `app.py` 冲突时，以 `main` 的最新契约者为准，再由两个模块负责人对齐签名。

## 五、版本与交付

- 包里 `__version__`（`kmeans_lab/__init__.py`）每次合并后由 PM 递增：`2.0.0 → 2.1.0`。
- 打包 exe 由 PM 在最终合并后执行（PyInstaller），产物走 Release，不接受个人提交 exe。
