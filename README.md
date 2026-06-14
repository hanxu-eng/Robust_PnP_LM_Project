# Robust_PnP_LM_Project

## 项目简介

本项目是优化课程大作业“计算软件类”的配套代码，题目为：

**面向三维场景重建的鲁棒 PnP 位姿优化模型与 Levenberg-Marquardt 算法实现**

项目实现了一个不依赖 OpenCV `solvePnP`、不调用 `scipy.optimize.least_squares` 的 PnP 位姿优化程序。核心算法为手写 Levenberg-Marquardt，实验对比普通 LM-PnP 与 Huber 鲁棒加权 LM-PnP 在不同噪声、外点比例、初始化误差、匹配点数量和 Huber 阈值条件下的表现。

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
│   ├── real_data.py
│   ├── real_experiment.py
│   ├── experiment.py
│   ├── plot_results.py
│   └── main.py
├── data/
│   └── .gitkeep
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

`main.py` 会运行一次固定随机种子的示例，并保存收敛曲线 `figures/convergence.svg`。

`quick` 模式用于快速验收，运行少量 trial，生成 CSV 结果和矢量图。

## 完整实验命令

```bash
python code/experiment.py --mode full
python code/plot_results.py
```

`full` 模式使用更多 trial 和更多参数设置，适合正式报告引用。建议服务器上优先运行 full 模式，然后用生成的 CSV 和 SVG 更新报告。

## 真实数据实验

项目支持读取 COLMAP sparse text model 的真实 2D-3D 对应关系。需要准备一个包含以下文件的目录：

```text
cameras.txt
images.txt
points3D.txt
```

建议优先使用 COLMAP undistorted sparse model 或 `PINHOLE` / `SIMPLE_PINHOLE` 相机模型；如果原模型包含径向畸变，当前脚本会忽略畸变参数，只使用等效 pinhole 内参做 PnP。

如果你手上是 COLMAP binary model，可以先转换成 text：

```bash
colmap model_converter \
  --input_path path/to/sparse/0 \
  --output_path data/colmap_text_model \
  --output_type TXT
```

然后运行真实数据实验：

```bash
python code/real_experiment.py --colmap-dir data/colmap_text_model --mode quick
```

正式运行：

```bash
python code/real_experiment.py --colmap-dir data/colmap_text_model --mode full
```

如果希望生成真实照片上的 overlay 可视化，需要同时传入原始图片目录：

```bash
python code/real_experiment.py \
  --colmap-dir data/colmap_text_model \
  --image-dir data/pipes/images/dslr_images_undistorted \
  --num-images 3 \
  --mode full \
  --format svg
```

默认情况下，脚本会自动选择 2D-3D 对应点最多的前 3 张图像分别运行 PnP 实验，并生成多图像汇总。若只想指定单张图，可继续使用 `--image-id` 或 `--image-name`。
`real_multi_image_montage.svg` 会将前三张选中图像横向拼成一行，便于直接放入报告正文。

若只想指定单张图像，可以使用：

```bash
python code/real_experiment.py --colmap-dir data/colmap_text_model --image-id 12
python code/real_experiment.py --colmap-dir data/colmap_text_model --image-name image001.jpg
```

真实数据实验会输出：

- `results/real_experiment_results.csv`
- `results/real_summary_results.csv`
- `results/real_per_image_summary_results.csv`
- `figures/real_reference_rmse.svg`
- `figures/real_rotation_error.svg`
- `figures/real_observed_rmse.svg`
- `figures/real_translation_error.svg`
- `figures/real_pose_dashboard.svg`
- `figures/real_per_image_reference_rmse.svg`
- `figures/real_per_image_rotation_error.svg`
- `figures/real_multi_image_montage.svg`
- `figures/real_keypoints_overlay_img*.svg`：真实照片上的 2D 观测点和 COLMAP reference 投影。
- `figures/real_reprojection_overlay_img*.svg`：错误匹配压力测试下的真实照片重投影 overlay。
- `figures/real_residual_vectors_img*.svg`：真实照片上的 Huber-LM 重投影残差向量。
- `figures/real_residual_histogram_img*.svg`：Ordinary-LM 与 Huber-LM 的残差分布对比。

多图像模式下，每张图像还会生成带图像编号的 overlay 文件，例如：

```text
real_keypoints_overlay_img002_DSC_0635.svg
real_reprojection_overlay_img002_DSC_0635.svg
real_residual_vectors_img002_DSC_0635.svg
real_residual_histogram_img002_DSC_0635.svg
```

说明：COLMAP 给出的位姿和 3D 点作为 reference pose / reference map。实验先在真实 2D-3D 对应关系上跑 PnP，再通过打乱一部分真实 2D 观测模拟错误匹配压力测试。这比纯合成点更接近真实视觉任务，但仍依赖 COLMAP 重建质量。

## 输出文件说明

运行实验后会生成：

- `results/experiment_results.csv`：每个 trial 的详细结果。
- `results/summary_results.csv`：按实验类型、参数设置和方法聚合后的均值与标准差。
- `figures/convergence.svg`：单次示例的 LM 收敛曲线。
- `figures/noise_clean_reprojection_rmse.svg`：噪声实验下的 clean 重投影 RMSE。
- `figures/noise_rotation_error.svg`：噪声实验下的旋转误差。
- `figures/outlier_clean_reprojection_rmse.svg`：外点实验下的 clean 重投影 RMSE。
- `figures/outlier_rotation_error.svg`：外点实验下的旋转误差。
- `figures/translation_error.svg`：外点实验下的平移误差。
- `figures/initialization_sensitivity.svg`：不同初始旋转扰动下的收敛表现。
- `figures/point_count_sensitivity.svg`：不同匹配点数量下的误差变化。
- `figures/huber_delta_sweep.svg`：Huber 阈值消融实验。
- `figures/outlier_clean_rmse_boxplot.svg`：外点实验逐 trial 误差分布箱线图。
- `figures/robustness_gain_heatmap.svg`：普通 LM 与 Huber-LM 的误差比热力图。
- `figures/performance_dashboard.svg`：四个主要设问的组合式总览图。
- `figures/real_reference_rmse.svg`：真实 COLMAP 对应关系上的参考重投影误差。
- `figures/real_rotation_error.svg`：真实 COLMAP 对应关系上的旋转误差。

默认图像格式为 SVG 矢量图。若需要 PDF，可运行：

```bash
python code/plot_results.py --format pdf
```

若需要同时生成 SVG 和 PDF，可运行：

```bash
python code/plot_results.py --format both
```

热力图使用矢量矩形 patch 绘制，避免在矢量文件中嵌入位图热力图。

## 实验设问

本项目的实验部分不只比较单一场景，而是围绕以下问题展开：

- **噪声敏感性**：图像观测高斯噪声增大时，两种方法的位姿误差如何变化。
- **外点鲁棒性**：错误匹配比例升高时，Huber 加权是否能稳定降低 clean 重投影误差。
- **初始化敏感性**：LM 是局部优化方法，初始位姿偏差增大后收敛质量是否下降。
- **匹配点数量影响**：在存在外点时，更多 3D-2D 对应点是否能提升估计稳定性。
- **Huber 阈值消融**：Huber `delta` 过小或过大时，鲁棒权重对最终精度有什么影响。

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

`experiment.py` 中记录了每次 trial 的初始化误差、点数量、噪声强度、外点比例、Huber 阈值、最终 cost 和成功标记，便于在报告中做更细的统计分析。

## 评价指标说明

实验数据中的 `observed_uv` 包含噪声和外点。如果直接用 `observed_uv` 评价，鲁棒方法可能因为主动降低外点权重而不一定在 observed error 上占优。因此本项目以 `clean_uv` 计算的 clean reprojection RMSE 作为主评价指标，用于衡量估计位姿相对真实投影的准确性；observed reprojection RMSE 作为辅助指标保留。

预期现象是：无外点或低外点情况下，Ordinary-LM 和 Huber-LM 表现接近；外点比例升高后，Ordinary-LM 的旋转误差、平移误差和 clean 重投影 RMSE 增长更明显，Huber-LM 平均更稳定。

## 注意事项

- 本项目不调用 OpenCV `solvePnP`。
- 本项目不调用 `scipy.optimize.least_squares`。
- 所有随机实验均设置 seed，结果可复现。
- 所有图由真实运行结果生成，不编造实验数值。
- quick 模式用于快速检查代码是否可运行；正式写报告建议使用 full 模式重新生成结果。
- 绘图优先使用 Matplotlib；若本地环境缺少 Matplotlib，代码会生成简单 SVG/PDF 兜底，服务器安装依赖后会自动生成更完整的高级图。
