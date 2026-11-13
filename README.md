# DoubleAdapt 多因子模型

简体中文 | [English](README.en.md)

本项目提供 DoubleAdapt 的独立社区实现，用于多因子股票收益预测、特征适配、标签适配和周期性信号推理。

模型包含基础 MLP、Feature Adapter 和 Label Adapter。Support 日期用于计算适配方向，Query 日期用于评价适配后的泛化表现。推理阶段只需要特征，不需要标签。

## 快速开始

```bash
python -m pip install -e .
python examples/make_sample.py
python scripts/validate_inputs.py --input examples/generated/train.parquet --config examples/train.yaml
python scripts/train.py --config examples/train.yaml --input examples/generated/train.parquet --output-dir outputs/demo
python scripts/infer.py --config examples/infer.yaml --checkpoint outputs/demo/checkpoint.pt --input examples/generated/test.parquet --output outputs/demo/signals.parquet
```

## 输入

训练面板至少包含：`date`、`ticker`、一个或多个 `feature_*` 列和 `label`。推理面板不需要 `label`。

## 输出

预测信号为 `date,ticker,prediction`，训练过程同时保存 checkpoint、训练历史和运行清单。

## 参考与边界

- [方法说明](references/method.md)
- [配置说明](references/configuration.md)
- [输入格式](references/input_schema.md)
- [输出格式](references/output_schema.md)
- [论文与实现声明](references/paper.md)

本项目为非官方社区实现，使用者需要自行提供数据并负责时间切分、标签成熟度和回测验证。仅供量化研究和教育用途。
