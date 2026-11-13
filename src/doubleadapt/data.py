from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

def validate_panel(frame: pd.DataFrame, require_label: bool = True) -> list[str]:
    required = {"date", "ticker"}
    if require_label:
        required.add("label")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame[["date", "ticker"]].duplicated().any():
        raise ValueError("date/ticker rows must be unique")
    feature_names = [c for c in frame.columns if c.startswith("feature_")]
    if not feature_names:
        raise ValueError("at least one feature_* column is required")
    values = frame[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    if require_label and not np.isfinite(frame["label"].to_numpy(dtype=float)).all():
        raise ValueError("label contains non-finite values")
    return feature_names

def load_panel(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    validate_panel(frame, require_label=require_label)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)

def to_tensor(frame: pd.DataFrame, features: Iterable[str]) -> tuple:
    import torch
    x = torch.tensor(frame[list(features)].to_numpy(dtype="float32"))
    y = torch.tensor(frame["label"].to_numpy(dtype="float32")) if "label" in frame else None
    return x, y
