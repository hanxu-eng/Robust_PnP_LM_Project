# 面向三维场景重建的鲁棒 PnP 位姿优化模型与 Levenberg-Marquardt 算法实现

报告类型：计算软件类

研究方向：计算机视觉中的三维场景重建

## 摘要

本文围绕三维场景重建中的相机位姿估计问题，设计并实现了一个基于 Levenberg-Marquardt 方法的 PnP 位姿优化程序。项目不调用 OpenCV `solvePnP` 和 `scipy.optimize.least_squares` 等现成求解器，而是从相机投影模型、重投影误差、数值雅可比和阻尼最小二乘方程出发，自行实现普通 LM-PnP 与 Huber 鲁棒加权 LM-PnP。实验使用可复现的合成数据，从噪声、外点、初始化误差、匹配点数量和 Huber 阈值多个角度分析算法表现。具体数值请从 `results/summary_results.csv` 中读取后填入。

## 1 引言

### 三维重建中的位姿估计问题

三维场景重建通常需要估计相机与场景之间的相对位姿。给定三维空间点及其二维图像观测，位姿估计可以恢复相机的旋转和平移，是三维重建、增强现实、机器人定位和视觉 SLAM 中的基础模块。

### PnP 问题的意义

PnP（Perspective-n-Point）问题研究的是在已知相机内参和若干三维-二维对应关系时估计相机位姿。由于重投影过程包含透视除法，PnP 的最小二乘形式通常是非线性且非凸的，需要使用迭代优化方法求解。

### 普通最小二乘对外点敏感

普通最小二乘将所有观测残差同等看待。当二维观测中存在错误匹配或严重偏离真实投影的外点时，大残差会主导目标函数，使估计位姿偏离真实值。因此，鲁棒损失或鲁棒加权策略对实际视觉任务具有重要意义。

### 本文工作

本文完成以下工作：

- 建立基于重投影误差的 PnP 位姿优化模型。
- 使用 Rodrigues 公式表示旋转变量。
- 自行实现数值雅可比和 Levenberg-Marquardt 迭代算法。
- 引入 Huber 权重，构造鲁棒加权 LM-PnP。
- 通过合成数据分析噪声、外点、初始化、点数量和 Huber 阈值对算法的影响。

## 2 问题建模

### 相机投影模型

设三维点为 \(X_i=[x_i,y_i,z_i]^T\)，相机位姿为旋转矩阵 \(R\) 和平移向量 \(t\)。相机坐标为：

```text
X_i^c = R X_i + t
```

在相机内参 \(K\) 下，投影坐标为：

```text
u_i = fx * X_i^c_x / X_i^c_z + cx
v_i = fy * X_i^c_y / X_i^c_z + cy
```

### 标准 PnP 重投影误差模型

令观测像素为 \(\hat{u}_i\)，投影函数为 \(\pi(\cdot)\)，标准最小二乘模型为：

```text
min_{r,t} sum_i || pi(K, R(r) X_i + t) - observed_uv_i ||^2
```

其中 \(r\) 为旋转向量，\(R(r)\) 由 Rodrigues 公式计算。

### Huber 鲁棒加权模型

为降低外点影响，Huber-LM 根据当前二维残差范数 \(\|e_i\|\) 计算权重：

```text
w_i = 1,               ||e_i|| <= delta
w_i = delta / ||e_i||, ||e_i|| > delta
```

然后优化加权重投影误差：

```text
min_{r,t} sum_i w_i ||e_i||^2
```

该模型在小残差区域接近普通最小二乘，在大残差区域降低异常点影响。

### 决策变量、约束、非凸性分析

决策变量为 6 维向量：

```text
params = [r_x, r_y, r_z, t_x, t_y, t_z]
```

其中前三维为旋转向量，后三维为平移向量。该模型没有显式等式或不等式约束，但由于透视投影中的除法和旋转参数化的非线性，目标函数是非凸的。因此 LM 算法依赖初始化。本项目在合成实验中使用真值附近的扰动初始化，用于模拟已有粗位姿或 EPnP 初始化后的局部优化阶段。

## 3 算法设计

### LM 方法基本思想

LM 方法可看作 Gauss-Newton 与梯度下降之间的折中。每次迭代构造线性化残差：

```text
r(params + delta) ≈ r(params) + J delta
```

并求解阻尼正规方程：

```text
(J^T J + lambda diag(J^T J)) delta = -J^T r
```

若更新降低目标函数，则接受更新并减小 `lambda`；否则拒绝更新并增大 `lambda`。

### 数值雅可比

项目使用中心差分计算雅可比：

```text
J[:, j] = (r(params + eps e_j) - r(params - eps e_j)) / (2 eps)
```

这样实现简单清晰，适合课程作业展示。

### Huber 权重

Huber-LM 在每次迭代根据当前二维残差计算点权重，并将每个点权重扩展到对应的 `u` 和 `v` 两个残差维度。加权残差和加权雅可比分别为：

```text
r_w = sqrt(W) r
J_w = sqrt(W) J
```

然后在加权空间内求解 LM 增量。

### 阻尼因子更新

实现中设置 `lambda` 的范围为 `[1e-12, 1e12]`。若候选位姿使 cost 下降，则接受更新并令 `lambda *= 0.5`；否则拒绝更新并令 `lambda *= 2.0`。若线性方程病态，则使用更大的阻尼或 `np.linalg.lstsq` 作为兜底。

### 算法伪代码

```text
Input: X, observed_uv, K, init_params
params = init_params
lambda = lambda_init

for iter = 1 ... max_iters:
    r = residual_vector(params)
    if robust:
        w = HuberWeights(r)
    else:
        w = 1

    J = NumericJacobian(params)
    rw = sqrt(w) * r
    Jw = sqrt(w) * J
    H = Jw.T @ Jw
    g = Jw.T @ rw

    solve (H + lambda diag(H)) delta = -g
    candidate = params + delta

    if cost(candidate) < cost(params):
        params = candidate
        lambda = lambda / 2
    else:
        lambda = lambda * 2

    if convergence criterion satisfied:
        break

Output: estimated rvec, t, cost history
```

## 4 仿真实验

### 合成数据设置

相机内参设置为：

```text
fx = 800, fy = 800, cx = 320, cy = 240
```

三维点坐标由均匀分布采样：

```text
x, y in [-2, 2]
z in [4, 8]
```

真实位姿为：

```text
rvec_gt = [0.15, -0.10, 0.08]
t_gt = [0.30, -0.20, 0.50]
```

观测点由 clean 投影叠加高斯噪声和外点生成。所有实验设置随机种子，保证可复现。

### 噪声实验

固定外点比例为 0.1，改变 `noise_sigma`。正式实验建议使用：

```text
noise_sigma_list = [0.5, 1.0, 2.0, 3.0]
```

报告中可引用：

- `figures/noise_clean_reprojection_rmse.svg`
- `figures/noise_rotation_error.svg`

具体数值从 `results/summary_results.csv` 中读取。

### 外点实验

固定高斯噪声为 1.0，改变外点比例。正式实验建议使用：

```text
outlier_ratio_list = [0.0, 0.1, 0.2, 0.3]
```

报告中可引用：

- `figures/outlier_clean_reprojection_rmse.svg`
- `figures/outlier_rotation_error.svg`
- `figures/translation_error.svg`

具体数值从 `results/summary_results.csv` 中读取。

### 初始化敏感性实验

LM 属于局部非线性优化方法，收敛效果依赖初始值。为体现非凸优化中的初始化问题，实验固定噪声和外点比例，改变初始旋转扰动幅度。正式实验建议使用：

```text
init_rot_deg_list = [2.0, 5.0, 10.0, 16.0, 24.0]
```

报告中可引用：

- `figures/initialization_sensitivity.svg`

该实验可用于讨论：初始扰动较小时两种方法都能收敛到较好解；扰动增大后，LM 可能落入较差局部区域。鲁棒权重主要缓解外点影响，但不能从根本上消除非凸优化的初始化依赖。

### 匹配点数量实验

在固定噪声和较高外点比例下，改变三维-二维对应点数量。正式实验建议使用：

```text
point_count_list = [30, 60, 120, 200]
```

报告中可引用：

- `figures/point_count_sensitivity.svg`

该实验可用于说明：更多匹配点通常能提高统计稳定性，但若外点比例固定，普通最小二乘仍可能被大残差主导；Huber-LM 对新增内点信息的利用更稳定。

### Huber 阈值消融实验

Huber 阈值 `delta` 控制残差从二次惩罚转入线性惩罚的边界。正式实验建议使用：

```text
huber_delta_list = [1.0, 2.0, 5.0, 10.0, 20.0]
```

报告中可引用：

- `figures/huber_delta_sweep.svg`

该实验可用于讨论：`delta` 过小可能过度压低正常噪声点权重，`delta` 过大则接近普通最小二乘，对外点抑制不足。

### 真实数据实验

为增强实验可信度，可以使用 COLMAP 在真实图像上得到的 sparse reconstruction。实验读取 `cameras.txt`、`images.txt` 和 `points3D.txt`，从单张图像中提取真实 2D-3D 对应关系，并将 COLMAP 位姿作为 reference pose。运行命令示例：

```text
python code/real_experiment.py --colmap-dir data/colmap_text_model --mode full
```

如果原始模型是 COLMAP binary 格式，可先执行：

```text
colmap model_converter --input_path path/to/sparse/0 --output_path data/colmap_text_model --output_type TXT
```

报告中可引用：

- `figures/real_reference_rmse.svg`
- `figures/real_rotation_error.svg`
- `figures/real_observed_rmse.svg`
- `figures/real_translation_error.svg`
- `figures/real_pose_dashboard.svg`
- `figures/real_keypoints_overlay.svg`
- `figures/real_reprojection_overlay.svg`
- `figures/real_residual_vectors.svg`
- `figures/real_residual_histogram.svg`

需要说明的是，真实数据实验中的 reference pose 不是独立的运动捕捉真值，而是 COLMAP 重建结果。该实验的意义在于验证算法能否在真实图像匹配产生的 2D-3D 对应关系上工作，并通过打乱部分真实 2D 观测模拟错误匹配压力测试。

### 评价指标

主要指标包括：

- clean reprojection RMSE
- observed reprojection RMSE
- rotation error degree
- translation error
- num_iters
- final_cost
- success_rate
- init_rot_deg
- huber_delta

其中 clean reprojection RMSE 使用无噪声、无外点的 `clean_uv` 作为真实投影来评价位姿质量。由于 `observed_uv` 包含外点，鲁棒方法会主动降低外点影响，因此 observed error 不一定总是更小；它更适合作为辅助观察指标。

### 图表说明

预期结果如下：

- 无外点或低外点情况下，Ordinary-LM 与 Huber-LM 表现接近。
- 随着外点比例升高，Ordinary-LM 的旋转误差、平移误差和 clean 重投影 RMSE 通常增长更明显。
- Huber-LM 通过降低大残差点权重，在高外点比例下平均更稳定。
- 初始化误差增大时，两种方法都会体现局部优化的收敛盆地限制。
- Huber 阈值存在折中：过大时鲁棒性减弱，过小时可能损失内点信息。
- 不要求 Huber-LM 在所有单次随机 trial 中都绝对更好，应重点观察平均结果。

建议报告中优先放入：

- `figures/performance_dashboard.svg`：总览噪声、外点、初始化和点数量四类实验。
- `figures/robustness_gain_heatmap.svg`：用误差比展示 Huber-LM 的鲁棒性收益。
- `figures/outlier_clean_rmse_boxplot.svg`：展示不同 trial 的分布，而不是只展示均值。

如果排版系统更偏好 PDF，可运行 `python code/plot_results.py --format pdf` 生成同名 PDF 矢量图。

## 5 结论与局限

### 总结贡献

本文从 PnP 重投影误差模型出发，实现了一个完整、可运行、可复现的 LM 位姿优化程序。通过普通 LM 与 Huber-LM 的对比，展示了鲁棒加权策略在外点存在时对三维重建位姿估计的稳定性提升。

### 局限

本项目仍存在以下局限：

- 实验数据为合成数据，尚未覆盖真实图像匹配误差。
- LM 属于局部优化方法，结果依赖初始化。
- 当前只优化单相机位姿，未扩展到完整 Bundle Adjustment。
- 雅可比使用数值差分，效率低于解析雅可比。
- Huber 阈值仍由人工设定，未实现自适应尺度估计。

## 参考文献建议

- Hartley, R., and Zisserman, A. *Multiple View Geometry in Computer Vision*.
- Lepetit, V., Moreno-Noguer, F., and Fua, P. EPnP: An Accurate O(n) Solution to the PnP Problem.
- Levenberg, K. A Method for the Solution of Certain Non-Linear Problems in Least Squares. 1944.
- Marquardt, D. An Algorithm for Least-Squares Estimation of Nonlinear Parameters. 1963.
- Huber, P. J. Robust Estimation of a Location Parameter. 1964.
- Triggs, B., McLauchlan, P., Hartley, R., and Fitzgibbon, A. Bundle Adjustment: A Modern Synthesis.
