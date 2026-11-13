from pathlib import Path
import numpy as np
import pandas as pd

out = Path("examples/generated")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
rows = []
for date in range(20240102, 20240107):
    for index in range(12):
        values = rng.normal(size=4)
        rows.append([date, f"{index:06d}.SZ", *values, float(values[0] * 0.2 + rng.normal())])
frame = pd.DataFrame(rows, columns=["date", "ticker", "feature_1", "feature_2", "feature_3", "feature_4", "label"])
frame.to_parquet(out / "train.parquet", index=False)
frame.loc[frame.date.eq(20240106), ["date", "ticker", "feature_1", "feature_2", "feature_3", "feature_4"]].to_parquet(out / "test.parquet", index=False)
print(out)
