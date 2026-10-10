# VaDE 批量聚类实验

Python 3.10 / PyTorch 的 Variational Deep Embedding 实现。
默认配置：全局 seed=42、batch=512、VaDE 500 epoch；先从头进行 100 epoch
确定性 AE 预训练，再用历史 GMM 初始化策略拟合潜空间并联合优化。
500 指 VaDE 阶段，AE 的 100 epoch 另外计入总训练和计时。

## 当前进展

- 已完成 Keras/Theano 到 PyTorch 的核心迁移，清理旧代码、作者权重和内置数据。
- 此前 MNIST 验证：使用作者 AE 权重重新初始化 GMM 并训练 3000 epoch，
  seed=42、batch=100，最终 ACC **94.1014%**，接近作者权重的 **94.4614%**。
  此结果属于历史验证，不是当前从头预训练、batch=512 的批量配置结果。
- 当前批量入口已配置 26 个数据集，数据文件均已就绪。
- 单数据集训练、批量调度和自动绘图已通过小规模验证，8 项测试通过。
- 正式批量实验已跑完 26 个数据集，统一配置、不进行超参数搜索。

## 结构

```text
vade/model.py        网络、混合先验、负 ELBO
vade/datasets.py     DC 数据格式解析、按数据集归一化与重构分布
vade/metrics.py      匈牙利对齐、NMI / ARI / macro F1
vade/experiment.py   单数据集训练与结果保存
scripts/run_batch.py 按指定顺序提交并行任务
tests/              数学及数据/指标测试
results/            新实验输出（不提交 Git）
```

历史作者代码/权重/内置数据和展示图片已移除；旧实验产物 `runs/` 保留作记录。
所有新实验独立从头训练，无需旧作者权重。

## 环境

```bash
conda env create -f environment.yml
conda activate vade
```

当前机器已安装 `vade` 环境。运行命令从项目根目录执行。

## 单个数据集

```bash
python -m vade spiral --device cuda:1
```

数据默认目录 `../data/DC`。MAT 使用 `data/class` 或 `X/Y`，MNIST 映射到
`mnist.mat`；空间 PKL 使用 `expression_normalized/ground_truth`，DLPFC 映射到
`151507_final.pkl`。归一化与重构分布按数据集类型选择（见下文），常数列归零。
保持原样本顺序，无标签依赖过滤。簇数使用数据集真值类别数，不使用真值优化模型。
空间坐标不参与训练。绘图使用归一化之前的原始特征（空间 PKL 为
expression_normalized，而不是再缩放后的训练特征），不使用模型潜空间表示。
参考 `../IDEC/idec_reproduction/plotting.py`：二维直接画原始坐标，高维用
t-SNE，seed=42、PCA 初始化、auto 学习率、perplexity=min(30,n-1)。
默认按真值比例分层抽样最多 5000 点，仅影响绘图，不影响训练/指标。
图片为 8×6 英寸、300 DPI、无坐标轴/标题，颜色使用 tab10/tab20/HSV。

### 归一化与重构分布

归一化和重构分布按数据集类型配套选择，由 `vade/datasets.py` 的
`reconstruction_for()` 决定，并记录到 `config.json`（`reconstruction`）与
`dataset.json`（`normalization`）：

| 数据集类型 | 数据集 | 归一化 | 重构分布 |
| --- | --- | --- | --- |
| 灰度像素图像 | MNIST、USPS、COIL20 | 逐特征 MinMax `[0,1]` | Bernoulli（BCE） |
| 连续型数据 | 其余 23 个 | 逐特征 z-score（StandardScaler） | Gaussian（MSE） |

- 灰度像素把强度视为 `[0,1]` 概率，按 VAE 惯例用 BCE；BCE 要求输入在 `[0,1]`，
  故配 MinMax。
- 连续型数据（2D 合成几何、高斯/稀疏点云、128 维深度特征、单细胞/空间转录组）
  用 z-score 保持连续量纲，MSE 重构项才有合理量级。若对这些数据仍用 MinMax，
  会压平量纲、使重构项远小于 KL，导致 GMM 坍缩成单簇（NMI 掉到 0）。
- 常数列在两种归一化下都归零；归一化只影响训练特征，绘图始终用归一化前的原始特征。

统一采用 500/500/2000 隐藏层、10 维潜变量和 alpha=1，重构分布按数据集类型选择（见上文）。
网络/GMM 初始学习率 .002，每 10 epoch 乘 .9，最低 .0002。
AE 学习率 .001，结束后 logvar 初始化为 -4；GMM 用 seed=42，10 次 KMeans
初始化均值，全局样本方差初始化分量方差，正则 1e-3、EM 最多 100 次。
这一统一配置用于跨数据集实验，不是每个数据集单独调参。

## 并行跑动

```bash
python -m scripts.run_batch --devices cuda:0 cuda:1
# 共享服务器上只用 GPU 1：
python -m scripts.run_batch --devices cuda:1
```

默认一个设备同时运行一个任务；任务按 `vade/datasets.py` 中用户指定的 26 个
数据集顺序提交，完成顺序取决于数据量。每个任务独立进程、独立 seed=42。
可用 `--datasets spiral AC` 指定子集，`--workers` 限制设备并行数。
已完成数据集默认跳过；`--overwrite` 重跑指定输出目录。
缺失数据会在对应目录写 `status.json`，其余数据继续执行。

## 输出

每个数据集保存到 `results/<dataset>/`：

- `run.log`：批量任务的全部训练日志与错误。
- `train.log`：单数据集入口也自动保存训练日志。
- `config.json`、`dataset.json`、`status.json`：参数、来源、运行状态。
- `history.jsonl`：AE/VaDE 逐 epoch 损失。
- `model.pt`：最终模型和优化器。
- `labels.npy`：原始簇 ID；`labels_aligned.npy`：对齐后的类别编码。
- `ground_truth.npy`：原始真值；`labels.csv`：样本索引、簇、对齐标签、真值编码。
- `alignment.json`：匈牙利映射和原始类别值。
- `normalization.npz`：归一化参数（MinMax 或 z-score，随重构分布而定）。
- `original_space_clustering.jpg`、`original_space_true_labels.jpg`：相同原始空间
  坐标上的最终聚类/真值图；聚类图使用对齐标签。
- `plot_data.npz`、`plot_config.json`：绘图索引、二维坐标、标签和投影配置。
- `metrics.json`、`metrics.csv`：NMI、ARI、macro F1 和 `time_seconds`。

三个指标在匈牙利对齐后计算，取值为比例（1.0000 表示满分）。
JSON 用固定四位小数的字符串保存，以保留末尾零；CSV 同样保存四位。
计时从数据加载开始，包含归一化、AE、GMM、VaDE 和最终预测，到标签产生结束；
排除进程启动/任务排队、指标计算与结果文件写入。GPU 计时边界显式同步。
绘图也在计时结束之后，不包含在 time_seconds 中。
并行实验耗时包含设备/CPU 竞争的影响。
批量结束生成 `results/batch_summary.json` 和按用户顺序排列的 `summary.csv`。

## 验证

```bash
python -m pytest -q
python -m scripts.run_batch --devices cpu --datasets spiral AC \
  --epochs 2 --pretrain-epochs 2 --results /tmp/opencode/vade-smoke
```

使用已保存的最终标签单独重新画图，无需重新训练：

```bash
python -m scripts.plot_results spiral --output-dir results/spiral
```
