# DoubleAdapt 多因子模型

简体中文 | [English](README.en.md)

本项目提供 DoubleAdapt 的独立社区实现，用于多因子股票收益预测、特征适配、标签适配、Support/Query 训练和周期性信号推理。

> 本项目是非官方社区实现。它不包含论文作者官方代码、私有数据或内部实验结果，也不承诺论文数值可直接复现。

## 适用场景

适合以下任务：

- 使用用户提供的多因子截面面板训练收益预测模型；
- 使用 Feature Adapter 调整输入特征表达；
- 使用 Label Adapter 调整成熟标签的尺度和偏移；
- 使用 Support 数据计算适配方向，用 Query 数据评价适配后的泛化表现；
- 冻结 checkpoint 后生成股票预测信号。

本仓库不负责因子挖掘、组合优化、交易执行或实时行情接入。

## 方法概览

```text
股票因子输入
    |
    v
Feature Adapter -> 基础 MLP -> 预测分数
    ^                         |
    |                         v
Support 标签 -> Label Adapter -> Query 评价
```

Support 和 Query 必须按时间顺序构造。Query 标签只用于评价或外层更新，不能回流到特征构造或训练日期之前。

## 安装

```bash
python -m pip install -e .
```

建议使用 Python 3.10 或更高版本。


## 方法简介

DoubleAdapt 面向非平稳股票预测任务，在基础预测网络之外加入两个适配模块：Feature Adapter 根据当前输入调整特征表达，Label Adapter 根据已成熟标签调整监督目标的尺度与偏移。训练时使用按时间切分的 Support/Query 任务，Support 用于计算适配方向，Query 用于评价适配后的泛化表现；推理阶段只使用特征和冻结的 checkpoint。

## 参考论文

本项目参考：

- **DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting**
- 论文链接：[ACM Digital Library](https://doi.org/10.1145/3580305.3599315)
- 预印本：[arXiv:2306.09862](https://arxiv.org/abs/2306.09862)

本仓库是独立的非官方社区实现，使用可配置的 MLP 预测骨干和适配器模块，不包含论文作者的官方代码。

## 快速开始

生成合成样例并运行训练、推理：

```bash
python examples/make_sample.py
python scripts/validate_inputs.py \\
  --input examples/generated/train.parquet \\
  --config examples/train.yaml
python scripts/train.py \\
  --config examples/train.yaml \\
  --input examples/generated/train.parquet \\
  --output-dir outputs/demo
python scripts/infer.py \\
  --checkpoint outputs/demo/checkpoint.pt \\
  --input examples/generated/test.parquet \\
  --output outputs/demo/signals.parquet
```

## 输入格式

训练面板至少包含：

| 字段 | 说明 |
| --- | --- |
| `date` | 有序时间标识 |
| `ticker` | 股票标识 |
| `feature_*` | 一个或多个因子列 |
| `label` | 训练标签 |

推理面板不需要 `label`。同一 `date,ticker` 只能出现一行，特征和标签必须是有限数值。

完整约束见 [输入格式](references/input_schema.md)。

## 配置与时间切分

训练配置包括：

- 基础 MLP 的层数、隐藏维度和 dropout；
- Feature/Label Adapter 的 head 数量和温度；
- 学习率、内层学习率、权重衰减和 epoch；
- train、validation 的日期集合。

建议先固定时间切分和随机种子，再比较模型参数。不要根据测试期结果选择 checkpoint 或适配周期。

## 输出

训练目录包含：

- `checkpoint.pt`：验证集指标最优模型；
- `history.json`：训练损失和验证集 Pearson IC；
- `selection_manifest.json`：checkpoint 选择记录。

推理输出为：

```text
date | ticker | prediction
```

输出格式见 [输出格式](references/output_schema.md)。

## 验证

```bash
python -m compileall -q src scripts
python -m pytest -q
node scripts/validate-qsh-form.mjs SKILL.md
```

测试覆盖模型输出维度和输入 schema。正式研究还应检查时间切分、标签成熟度、特征快照和回测口径。

## 研究边界

本项目只生成股票横截面预测信号，不提供收益保证，也不构成投资建议。使用者需要自行确认点时数据、标签成熟度、交易成本、股票池和回测规则。

## 许可证

GPL-3.0-only，见 [LICENSE](LICENSE)。
