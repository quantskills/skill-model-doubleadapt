# DoubleAdapt Multi-Factor Model

[简体中文](README.md)

This repository provides an independent community implementation of DoubleAdapt for cross-sectional stock return prediction, feature adaptation, label adaptation, Support/Query training, and periodic signal inference.

The model combines a base MLP with Feature Adapter and Label Adapter modules. Support dates provide the adaptation direction, while Query dates evaluate generalization after adaptation.

> This is an unofficial community implementation. It does not include the paper authors' code, private data, or internal experiment outputs, and it does not claim numerical reproduction of the paper.

## Scope

The project supports user-supplied factor panels, chronological splits, validation checkpoint selection, adapter training, and portable signal export. It does not provide factor mining, portfolio optimization, execution, or live market data access.

## Installation

```bash
python -m pip install -e .
```

## Quick start

```bash
python examples/make_sample.py
python scripts/validate_inputs.py --input examples/generated/train.parquet --config examples/train.yaml
python scripts/train.py --config examples/train.yaml --input examples/generated/train.parquet --output-dir outputs/demo
python scripts/infer.py --checkpoint outputs/demo/checkpoint.pt --input examples/generated/test.parquet --output outputs/demo/signals.parquet
```

## Data contract

Training data must contain `date`, `ticker`, one or more `feature_*` columns, and `label`. Inference data omits `label`. Keys must be unique and numeric values must be finite.

Support and Query samples must be ordered in time. Query labels are used only for evaluation or outer updates and must not be used to construct earlier features or training labels.

See the [method](references/method.md), [configuration](references/configuration.md), [input schema](references/input_schema.md), and [output schema](references/output_schema.md).

## Validation

```bash
python -m compileall -q src scripts
python -m pytest -q
node scripts/validate-qsh-form.mjs SKILL.md
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).

## Method overview

DoubleAdapt targets non-stationary stock prediction by combining a base forecasting network with two adapters. The Feature Adapter adjusts the input representation, while the Label Adapter adjusts the scale and offset of mature supervision labels. Time-ordered Support/Query tasks are used for adaptation: Support data provides the adaptation direction and Query data evaluates post-adaptation generalization. Inference uses features and a frozen checkpoint only.

## Reference paper

This project is inspired by **DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting**.

- [ACM Digital Library](https://doi.org/10.1145/3580305.3599315)
- [arXiv:2306.09862](https://arxiv.org/abs/2306.09862)

This repository is an independent, unofficial community implementation and does not include the authors' official code.
