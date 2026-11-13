# 输入格式

训练面板为 Parquet 或 CSV，至少包含 `date`、`ticker`、一个或多个 `feature_*` 列和 `label`。推理面板不需要 `label`。同一 `date,ticker` 只能出现一行，特征必须是有限数值。
