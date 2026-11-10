# K-Means 聚类算法迭代过程可视化

## 一、项目简介

本项目是“拼好组”为《计算机与专业基础研讨》大作业开发的桌面演示程序。程序使用 Python 的 Tkinter 构建界面、Matplotlib 绘制图形，完整模拟 K-Means 聚类算法在二维数据上的迭代过程：散点按所属簇着色，质心随迭代移动，SSE 与轮廓系数实时更新，帮助学习者直观理解无监督学习中的聚类机制。

## 二、安装与运行指南

### 1. 环境依赖

- 操作系统：Windows 10/11、macOS 或 Linux
- Python 版本：Python 3.8 及以上
- 第三方库：NumPy、Matplotlib（Tkinter 随 Python 自带，无需单独安装）

```
pip install numpy matplotlib
```

### 2. 运行步骤

打开终端（Terminal）或命令提示符，执行以下命令：

```
# 1. 克隆项目到本地
git clone https://github.com/hy-123321/kmeans-visualizer.git

# 2. 进入项目目录
cd kmeans-visualizer

# 3. 安装依赖
pip install numpy matplotlib

# 4. 运行主程序
python K-means.py

# 5. 不弹窗口，只跑单元测试（21 项）
python K-means.py --self-test
```

## 三、项目结构与模块职责

程序按职责划分为若干模块，各模块只通过固定接口交互，便于独立开发和单元测试：

```
K-means.py                 程序入口（保持 python K-means.py 的用法）
kmeans_lab/
  config.py                全局常量、类型别名、数据集清单
  algorithm.py             KMeans2D 状态机：分配簇、更新质心、收敛判断
  initialization.py        质心初始化：随机选点与 K-means++
  data.py                  四种数据形状生成（三簇 / 四簇 / 同心圆 / 笑脸）
  metrics.py               SSE、轮廓系数、K 值推荐
  view.py                  Matplotlib 绘制：散点、置信椭圆、质心、K 值分析图
  ui.py                    Tkinter 界面布局与控件样式
  animation.py             单步补间的帧控制与连续运行调度
  export.py                聚类结果 CSV 与可视化图像导出
  app.py                   控制器：界面状态与各模块的调用关系
  main.py                  命令行入口与测试调度
tests/                     单元测试（21 项，覆盖数据、算法、指标、绘制与导出）
docs/
  MANUAL.md                操作手册
  AI_Prompts_Log.md        AI 协作记录
```

数据流为：`app` 读取界面设置 → `data` 生成数据 → `algorithm` 迭代并调用 `metrics` 计算指标 → `view` 按当前状态重绘 → `animation` 控制重绘的节奏。算法层不依赖任何界面代码，因此可以脱离图形界面单独测试。

## 四、GitHub 仓库链接与 Git 协作记录

仓库地址：https://github.com/hy-123321/kmeans-visualizer

访问权限：公开（Public）

### 分支记录（Branch History）：

- **main**：主分支，存放最终发布的稳定版本。
- **docs/requirements**：需求与设计文档编写分支。

开发过程中，各功能模块在独立分支上完成后再合并回 `main`，避免多人同时修改互相覆盖。

### 核心提交记录（Commit History）节选：

- b90874c [袁恒宇] init: 初始化仓库结构，添加 .gitignore 和 README
- 555785e [庞照阳] feat(ui): 实现主窗口布局与 Canvas 画布初始化
- fc66950 [张迎奥] feat(core): 实现 K-Means 核心迭代逻辑
- 89d3581 [叶世航] feat(animation): 实现单步执行与自动播放控制
- b73151b [袁恒宇] refactor: 按模块结构拆分代码，补充测试与文档

## 五、软件运行截图

<img width="2476" height="1634" alt="K-means可视化程序截图" src="https://github.com/user-attachments/assets/9b438744-9166-4be5-9040-6c03a40313b6" />

## 六、主要功能说明

- **数据生成**：内置三簇高斯、四簇高斯、同心圆、笑脸轮廓四种数据形状，可调节样本数（60–500）、随机种子与数据离散程度，相同参数可复现同一批数据。
- **簇数 K**：支持 K = 2–8，切换后立即重新聚类。
- **质心初始化**：支持 K-means++ 与随机选点两种方式，也可在图上依次点击 K 个位置手动指定初始质心（Esc 取消）。
- **单步迭代 / 连续运行**：单步执行一轮“分配簇 + 更新质心”，质心以补间动画移动到新位置；连续运行按固定节奏自动迭代，可随时暂停。
- **实时可视化**：散点按簇着色，质心以黑色叉号标记，彩色虚线为各簇的 95% 置信椭圆（按协方差矩阵计算），箭头表示本轮质心从旧位置到新位置的移动方向。
- **指标面板**：实时显示迭代轮数、簇内平方和（SSE）、轮廓系数与有效簇数；状态栏给出 SSE 相对初始状态的下降比例。
- **结果面板**：列出每个簇的样本数与质心坐标。
- **K 值分析**：对 K = 2–8 分别聚类至收敛，绘制 SSE 肘部曲线与轮廓系数曲线，自动标注推荐 K 值并给出综合建议。
- **结果导出**：导出每个样本的坐标、所属簇与对应质心为 CSV；可将当前可视化保存为 PNG / PDF / SVG / EPS 图像。

## 七、贡献者名单

本程序由“K-Means 可视化小组”共同开发：

- **袁恒宇**（项目经理 PM）：统筹进度与规范、核心算法、代码合并与 README 编写
- **庞照阳**（UI/UX 交互设计师）：系统 UI 布局、Canvas 绘制与操作手册编写
- **张迎奥**（算法工程师）：数据生成、质心初始化与 SSE 计算
- **叶世航**（动画与文档）：动画控制、AI_Prompts_Log 记录与测试

## 八、开源协议

本项目采用 MIT License 开源协议。欢迎将其用于计算机科学的教学与展示！
