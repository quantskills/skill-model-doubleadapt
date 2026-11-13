from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from .model import DoubleAdaptModel
from .data import validate_panel

def load_checkpoint(path: str | Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    config = ck["config"]
    model = DoubleAdaptModel(ck["input_size"], config["hidden_size"], config["layers"],
                             config["dropout"], config["heads"], config["temperature"])
    model.load_state_dict(ck["model_state"]); model.eval()
    return model, ck

def predict(model, frame: pd.DataFrame, feature_names: list[str], device: str = "cpu") -> pd.DataFrame:
    validate_panel(frame, require_label=False)
    x = torch.tensor(frame[feature_names].to_numpy("float32"), device=device)
    with torch.no_grad(): pred = model(x).cpu().numpy()
    return pd.DataFrame({"date": frame["date"].to_numpy(), "ticker": frame["ticker"].to_numpy(), "prediction": pred})
