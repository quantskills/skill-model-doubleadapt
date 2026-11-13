---
name: model-doubleadapt
description: Use when training a standalone DoubleAdapt cross-sectional stock model on a user-supplied multi-factor panel, adapting features and labels through Support/Query updates and exporting portable stock signals.
quantSkills:
  schema_version: 2.0.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-model-doubleadapt
  repository_url: https://github.com/quantskills/skill-model-doubleadapt
  project_type: skill
  license: GPL-3.0-only
  maintainer: X-Tech-group
  collection: research-models
  catalog: {category: "06", subcategory: 06.statistical-ml-models}
  workflow: {primary_stage: modeling, workflow_stages: [modeling]}
  tags: [model-doubleadapt, factor-model, alpha-model, online-adaptation, support-query, signal-export]
  requires: []
  summary_zh: 使用 DoubleAdapt 多因子模型进行特征适配、标签适配和股票横截面信号生成。
  summary_en: Trains DoubleAdapt on a supplied factor panel and exports cross-sectional stock signals.
  status: active
  validation_level: runnable
  maintainer_type: community
  platforms: [cursor, claude-code, codex, hermes, openclaw]
  interface:
    mode: structured
    envelope: {name: quantskills-envelope, version: 1.0.0}
    inputs:
      - {profile: factor-panel, version_range: ">=1.0.0 <2.0.0", required: true}
    outputs:
      - {profile: factor-signal, version: 1.0.0}
      - {profile: model-run-manifest, version: 1.0.0}
    adapters: [factor-model-training, factor-signal-export]

---

```json qsh-form
{
  "version": 1,
  "task": {"placeholder": "提供多因子面板、标签定义、时间切分与输出目录", "required": true},
  "fields": [
    {"key": "run_mode", "label": "运行模式", "type": "select", "options": [{"value": "train", "label": "训练"}, {"value": "predict", "label": "导出信号"}]}
  ],
  "prompt_template": "{{task}}\n\n使用 DoubleAdapt 执行 {{run_mode}}。先校验面板和时间切分，再运行对应命令并检查输出清单。"
}
```

# DoubleAdapt 多因子模型

这个 Skill 提供 DoubleAdapt 的独立社区实现，用于多因子股票收益预测、特征适配、标签适配和周期性信号推理。

## 使用范围

- 训练基础 MLP 预测未来收益；
- 使用 Feature Adapter 调整当前输入特征；
- 使用 Label Adapter 调整成熟标签的尺度与偏移；
- 使用 Support/Query 机制训练和评价适配器；
- 按固定周期更新适配器并生成预测信号；
- 输出标准化的 `date | ticker | prediction` 信号文件。

模型不包含因子挖掘、组合优化、实时交易或具体市场数据。用户需要提供自己的因子面板和标签。

## 主要入口

```bash
python scripts/validate_inputs.py --input /path/to/panel.parquet --config examples/train.yaml
python scripts/train.py --config examples/train.yaml --input /path/to/panel.parquet --output-dir outputs/run_001
python scripts/infer.py --config examples/infer.yaml --checkpoint outputs/run_001/checkpoint.pt --input /path/to/features.parquet --output outputs/signals.parquet
```

## 关键约束

- 训练、验证和推理日期必须由配置显式指定；
- Query 标签只能用于评价适配后的泛化表现；
- 推理阶段不要求标签列，也不读取未来标签；
- 日期与股票代码必须唯一；
- 运行清单记录输入字段、配置、checkpoint 和输出文件；
- 输出仅供量化研究和教育用途，不构成投资建议。

详细内容见 `references/` 下的方法、配置、输入输出和论文说明。

## 社区声明

本项目是参考 DoubleAdapt 论文方法和公开描述的非官方社区实现，不包含作者官方代码或内部实验数据。
