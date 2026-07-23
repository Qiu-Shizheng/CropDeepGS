# CropDeepGS 中文使用说明

CropDeepGS 是一个面向作物全基因组预测的 Python 深度学习工具包。它可以读取
SNP 剂量、数量性状和可选的实测环境变量，并使用统一的命令行流程分析不同作物。

[English documentation](README.md)

## 主要功能

- 同时学习加性遗传效应、非线性遗传效应和基因型与环境互作。
- 支持仅使用基因型，或联合使用基因型和环境变量。
- 支持五折交叉验证，并保证同一遗传材料不会同时进入训练集和测试集。
- 支持多年数据的留一年验证。
- 每个训练划分内独立完成缺失值填补、标准化、分类变量编码和 SNP 降维。
- 输出每个样本的预测值、每个划分的指标、汇总指标和运行参数。
- 支持 CPU 和 CUDA GPU。

## 环境要求

- Python 3.9 或更高版本
- NumPy 1.23 或更高版本
- pandas 1.5 或更高版本
- scikit-learn 1.2 或更高版本
- PyTorch 2.0 或更高版本

仓库内置示例可直接使用 CPU 运行。数据量较大时建议使用 CUDA GPU。

## 安装

```bash
git clone https://github.com/Qiu-Shizheng/CropDeepGS.git
cd CropDeepGS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

检查是否安装成功：

```bash
cropdeepgs --help
```

## 输入数据

CropDeepGS 需要两个 CSV 或 TSV 表格。

### 基因型表

第一列通常命名为 `sample_id`，表示材料编号。其余各列为数值型 SNP
剂量，通常编码为 `0`、`1` 和 `2`。

```text
sample_id  snp_0001  snp_0002  snp_0003
line_001   0         1         2
line_002   2         1         0
```

每个材料在基因型表中只能出现一次。SNP 可以含有缺失值，程序会在每个训练
划分中计算填补值，再将其应用于相应测试集。

### 表型表

表型表应包含材料编号、待预测性状和可选的实测环境变量。同一材料在多个环境中
接受测定时，可以在表型表中出现多行。

```text
sample_id  yield  year  soil_n  rain_mm  irrigation  line_group
line_001   5.42   2022  0.18    126.4    standard    line_001
line_002   4.91   2022  0.11    109.8    low         line_002
```

环境变量可以是数值型或分类型，例如土壤性质、天气统计量和管理措施。只能使用
实际预测时能够获得的变量，不能将目标表型或由目标表型计算得到的变量作为输入。

常用列含义：

| 列名 | 含义 |
|---|---|
| `sample_id` | 与基因型表对应的材料编号 |
| `yield` | 示例中的待预测产量，可替换为其他数量性状 |
| `year` | 仅用于定义留一年验证的年份 |
| `soil_n` | 示例土壤氮含量 |
| `rain_mm` | 示例降雨量 |
| `irrigation` | 示例灌溉处理 |
| `line_group` | 用于划分交叉验证的遗传材料或家系编号 |

## 快速开始

运行仓库内置的完整示例：

```bash
bash examples/run_example.sh
```

该脚本会重新生成一个小型模拟数据集，并依次执行五折交叉验证和留一年验证。

分析自己的数据：

```bash
cropdeepgs \
  --genotype path/to/genotypes.tsv \
  --phenotype path/to/phenotypes.tsv \
  --trait yield \
  --sample-col sample_id \
  --env-cols soil_n,rain_mm,irrigation \
  --group-col line_group \
  --year-col year \
  --eval fivefold,leave-year \
  --out results/yield
```

如果没有环境变量，只需省略 `--env-cols`：

```bash
cropdeepgs \
  --genotype path/to/genotypes.tsv \
  --phenotype path/to/phenotypes.tsv \
  --trait protein_content \
  --group-col line_group \
  --eval fivefold \
  --out results/protein_content
```

## 验证方法

### 五折交叉验证

使用 `--eval fivefold`。程序将遗传材料分成五组，每次使用其中一组作为测试集，
其余四组作为训练集。建议通过 `--group-col` 指定材料、品系、家系或其他遗传
分组列；如果不指定，程序使用 `sample_id`。同一分组中的记录始终位于同一折，
可避免同一材料的重复观测同时进入训练集和测试集。

### 留一年验证

使用 `--eval leave-year --year-col year`。程序每次将一年的全部记录作为测试集，
其余年份作为训练集。缺失值填补、标准化、分类变量编码和 SNP 降维均只使用
训练年份拟合。训练记录不少于 80 条且测试记录不少于 20 条时才建立该年份的
验证任务。年份列只负责划分数据，不会自动作为模型输入。

两种验证可同时执行：

```bash
--eval fivefold,leave-year
```

## 输出结果

输出目录包含：

| 文件 | 内容 |
|---|---|
| `predictions.tsv` | 每条测试记录的实测值和预测值 |
| `split_metrics.tsv` | 每一折或每一年的全部评价指标 |
| `summary_metrics.tsv` | 各验证方法的平均指标 |
| `run_config.json` | 输入列、验证设置和模型参数 |

评价指标解释：

- `pearson`：皮尔逊相关系数，越高越好，反映预测值与实测值的线性一致性。
- `spearman`：斯皮尔曼等级相关系数，越高越好，反映材料排序的一致性。
- `rmse`：均方根误差，越低越好，对较大的预测误差更敏感。
- `mae`：平均绝对误差，越低越好。
- `r2`：决定系数，越高越好；测试数据上可能小于 0。
- `nrmsep`：标准化均方根预测误差，等于 RMSE 除以测试集表型标准差，越低越好。

## 主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--snp-pcs` | `512` | 每个训练划分保留的最大 SNP 主成分数 |
| `--hidden` | `384` | 神经网络隐藏层宽度 |
| `--dropout` | `0.22` | Dropout 比例 |
| `--epochs` | `152` | 最大训练轮数 |
| `--patience` | `18` | 提前停止等待轮数 |
| `--batch-size` | `1024` | 训练批次大小 |
| `--lr` | `0.00035` | 学习率 |
| `--seed` | `47` | 随机种子 |
| `--device` | 自动选择 | 可指定 `cuda`、`cuda:0` 或 `cpu` |

运行 `cropdeepgs --help` 可查看全部高级参数。

## 计算资源

内存主要由样本数、SNP 数和保留的主成分数决定。将稠密 SNP 表读入内存时，
原始数值数组大约需要 `8 × 样本数 × SNP 数` 字节，pandas 和
scikit-learn 还会产生额外开销。超大数据应先完成样本和 SNP 质量控制，再生成
数值型矩阵。GPU 显存不足时，可依次减小 `--batch-size`、`--hidden` 和
`--snp-pcs`。

## 结果复现

随机种子和模型参数会写入 `run_config.json`。比较多个模型时，应保持验证划分、
表型记录和输入变量完全一致。不同 CUDA 硬件和软件版本之间可能存在很小的数值
差异。

## 目录结构

```text
cropdeepgs/                 Python 工具包
examples/                   模拟输入数据和可运行示例
tests/                      工具包基础测试
README.md                   英文说明
README_zh.md                中文说明
```

## 许可证

CropDeepGS 使用 MIT License 发布。
