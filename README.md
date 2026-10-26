# K-Means 聚类算法迭代过程可视化

## 一、项目简介

本项目是“拼好组”为《计算机与专业基础研讨》大作业开发的桌面演示程序。本项目通过 Python 的 Tkinter 图形库，完整模拟了 K-Means 聚类算法在二维数据上的迭代过程。通过可视化的散点分布、质心移动与实时运行日志，帮助学习者直观理解无监督学习中的聚类机制。

## 二、安装与运行指南

### 1. 环境依赖

- 操作系统：Windows 10/11、macOS 或 Linux
- Python 版本：Python 3.8 及以上版本（自带 tkinter，无需额外使用 pip 安装）

### 2. 运行步骤

打开终端（Terminal）或命令提示符，执行以下命令：

```
# 1. 克隆项目到本地
git clone https://github.com/hy-123321/kmeans-visualizer.git

# 2. 进入项目目录
cd kmeans-visualizer

# 3. 运行主程序
python K-means.py
```

## 三、GitHub 仓库链接与 Git 协作记录

仓库地址：https://github.com/hy-123321/kmeans-visualizer

访问权限：公开（Public）

### 分支记录（Branch History）：

- **main**：主分支，存放最终发布的稳定版本。
- **docs/requirements**：文档编写分支。
- **feature/ui**：Tkinter 界面布局与 Canvas 绘制分支。
- **feature/kmeans-core**：K-Means 核心迭代逻辑分支。
- **feature/animation**：动画控制与交互分支。

### 核心提交记录（Commit History）与冲突解决（PR）：

- b90874c [袁恒宇] init: 初始化仓库结构，添加 .gitignore 和 README
- 1eef19c [袁恒宇] docs: 提交《项目需求与设计文档》初稿
- ca78e9d [袁恒宇] docs: 补充 README 项目说明
- 555785e [庞照阳] feat(ui): 实现主窗口布局与 Canvas 画布初始化
- fc66950 [张迎奥] feat(core): 实现 K-Means 核心迭代逻辑
- 89d3581 [叶世航] feat(animation): 实现单步执行与自动播放控制
- a5ff33a [叶世航] docs: 完善最终版 README 与项目结题总结

## 四、软件运行截图

<img width="2476" height="1634" alt="K-means可视化程序截图" src="https://github.com/user-attachments/assets/9b438744-9166-4be5-9040-6c03a40313b6" />


## 五、主要功能说明

- **自定义K值**：可设置数据点数量与聚类数 K（2–6），点击“生成数据”随机撒点。
- **单步执行**：点击“单步执行”按钮，算法执行一次“分配簇”或“更新质心”操作。
- **自动播放**：点击“自动播放”按钮，系统按设定速度自动连续迭代。
- **实时可视化**：散点按簇着色，质心用叉形标记，可显示点到质心的归属连线。
- **统计面板**：实时显示当前迭代轮数、质心坐标、SSE 与收敛状态。

## 六、贡献者名单

本程序由“K-Means 可视化小组”共同开发：

- **袁恒宇**（项目经理 PM）：统筹进度与规范、核心算法、代码合并与 README 编写
- **庞照阳**（UI/UX 交互设计师）：系统 UI 布局、Canvas 绘制与操作手册编写
- **张迎奥**（算法工程师）：数据生成、质心初始化与 SSE 计算
- **叶世航**（动画与文档）：动画控制、AI_Prompts_Log 记录与测试

## 七、开源协议

本项目采用 MIT License 开源协议。欢迎将其用于计算机科学的教学与展示！
