# Robust_PnP_LM_Project

## 项目简介

本项目是优化课程大作业“计算软件类”的配套代码，题目为：

**面向三维场景重建的鲁棒 PnP 位姿优化模型与 Levenberg-Marquardt 算法实现**

项目实现了一个不依赖 OpenCV `solvePnP`、不调用 `scipy.optimize.least_squares` 的 PnP 位姿优化程序。核心算法为手写 Levenberg-Marquardt，实验对比普通 LM-PnP 与 Huber 鲁棒加权 LM-PnP 在不同噪声和外点比例下的表现。

由于 PnP 是非凸问题，LM 属于局部优化方法，本项目在合成实验中使用真值附近的扰动初始化，用于模拟已有粗位姿或 EPnP 初始化后的局部优化阶段。

## 文件结构

```text
Robust_PnP_LM_Project/
├── README.md
├── requirements.txt
├── AGENTS.md
├── code/
│   ├── __init__.py
│   ├── utils.py
│   ├── pnp_lm.py
│   ├── experiment.py
│   ├── plot_results.py
│   └── main.py
├── results/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
└── report_materials/
    └── report_outline.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

项目只需要：

- `numpy`
- `matplotlib`

## 快速运行命令

```bash
python code/main.py
python code/experiment.py --mode quick
python code/plot_results.py
```

`main.py` 会运行一次固定随机种子的示例，并保存收敛曲线 `figures/convergence.pdf`。

`quick` 模式用于快速验收，运行少量 trial，生成 CSV 结果和 PDF 图。

## 完整实验命令

```bash
python code/experiment.py --mode full
python code/plot_results.py
```

`full` 模式使用更多 trial 和更多参数设置，适合正式报告引用。

## 输出文件说明

运行实验后会生成：

- `results/experiment_results.csv`：每个 trial 的详细结果。
- `results/summary_results.csv`：按实验类型、参数设置和方法聚合后的均值与标准差。
- `figures/convergence.pdf`：单次示例的 LM 收敛曲线。
- `figures/noise_clean_reprojection_rmse.pdf`：噪声实验下的 clean 重投影 RMSE。
- `figures/noise_rotation_error.pdf`：噪声实验下的旋转误差。
- `figures/outlier_clean_reprojection_rmse.pdf`：外点实验下的 clean 重投影 RMSE。
- `figures/outlier_rotation_error.pdf`：外点实验下的旋转误差。
- `figures/translation_error.pdf`：外点实验下的平移误差。

所有图均保存为 PDF 矢量图。

## 算法说明

PnP 位姿估计以旋转向量 `rvec` 和平移向量 `t` 为决策变量，最小化三维点投影到图像后的重投影残差：

```text
min_{r,t} sum_i || pi(K, R(r) X_i + t) - u_i ||^2
```

其中 `R(r)` 由 Rodrigues 公式得到。LM 算法每次迭代通过数值雅可比构造正规方程：

```text
(J^T J + lambda diag(J^T J)) delta = -J^T r
```

若候选更新降低 cost，则接受更新并减小阻尼因子；否则拒绝更新并增大阻尼因子。

Huber-LM 使用当前二维残差范数计算点权重：

```text
w_i = 1,                  ||r_i|| <= delta
w_i = delta / ||r_i||,    ||r_i|| > delta
```

再将权重扩展到二维残差和雅可比上，从而降低大残差外点对优化的影响。

## 评价指标说明

实验数据中的 `observed_uv` 包含噪声和外点。如果直接用 `observed_uv` 评价，鲁棒方法可能因为主动降低外点权重而不一定在 observed error 上占优。因此本项目以 `clean_uv` 计算的 clean reprojection RMSE 作为主评价指标，用于衡量估计位姿相对真实投影的准确性；observed reprojection RMSE 作为辅助指标保留。

预期现象是：无外点或低外点情况下，Ordinary-LM 和 Huber-LM 表现接近；外点比例升高后，Ordinary-LM 的旋转误差、平移误差和 clean 重投影 RMSE 增长更明显，Huber-LM 平均更稳定。

## 注意事项

- 本项目不调用 OpenCV `solvePnP`。
- 本项目不调用 `scipy.optimize.least_squares`。
- 所有随机实验均设置 seed，结果可复现。
- 所有图由真实运行结果生成，不编造实验数值。
- quick 模式用于快速检查代码是否可运行；正式写报告建议使用 full 模式重新生成结果。
