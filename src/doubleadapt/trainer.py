from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .model import DoubleAdaptModel

def pearson_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().float(); target = target.detach().float()
    pred = pred - pred.mean(); target = target - target.mean()
    denom = torch.sqrt((pred.square().sum()) * (target.square().sum())).item()
    return float((pred * target).sum().item() / denom) if denom > 0 else float("nan")

def support_query_step(model, optimizer, support_x, support_y, query_x, query_y, inner_lr=1e-2):
    model.train()
    support_pred = model(support_x)
    adapted_support = model.adapted_label(support_x, support_y)
    support_loss = (support_pred - adapted_support).square().mean()
    named = dict(model.named_parameters())
    adapter_names = [name for name in named if name.startswith(("feature_adapter.", "label_adapter."))]
    params = [named[name] for name in adapter_names]
    grads = torch.autograd.grad(support_loss, params, create_graph=True, allow_unused=True)
    fast = dict(named)
    for name, param, grad in zip(adapter_names, params, grads):
        fast[name] = param - inner_lr * (grad if grad is not None else torch.zeros_like(param))
    from torch.func import functional_call
    query_pred = functional_call(model, fast, (query_x,))
    query_target = model.adapted_label(query_x, query_y)
    query_loss = (query_pred - query_target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    query_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(query_loss.detach()), pearson_score(query_pred, query_y)

def train_model(model, train_dates, valid_dates, frame, config, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    features = [c for c in frame.columns if c.startswith("feature_")]
    date_groups = {date: group for date, group in frame.groupby("date", sort=True)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best = -float("inf"); history = []
    train_dates = [d for d in train_dates if d in date_groups]
    for epoch in range(int(config["epochs"])):
        model.train(); losses = []; scores = []
        for left, right in zip(train_dates[:-1], train_dates[1:]):
            if left not in date_groups or right not in date_groups: continue
            a, b = date_groups[left], date_groups[right]
            sx, sy = torch.tensor(a[features].to_numpy("float32")), torch.tensor(a.label.to_numpy("float32"))
            qx, qy = torch.tensor(b[features].to_numpy("float32")), torch.tensor(b.label.to_numpy("float32"))
            loss, score = support_query_step(model, optimizer, sx, sy, qx, qy, config["inner_lr"])
            losses.append(loss); scores.append(score)
        valid_scores = []
        model.eval()
        with torch.no_grad():
            for date in valid_dates:
                if date not in date_groups: continue
                group = date_groups[date]
                x = torch.tensor(group[features].to_numpy("float32")); y = torch.tensor(group.label.to_numpy("float32"))
                valid_scores.append(pearson_score(model(x), y))
        valid = float(np.nanmean(valid_scores)) if valid_scores else float("nan")
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None, "valid_pearson": valid}
        history.append(row)
        if np.isfinite(valid) and valid > best:
            best = valid
            torch.save({"model_state": model.state_dict(), "input_size": len(features), "feature_names": features,
                        "config": config, "best_epoch": epoch + 1, "best_valid_pearson": best}, output / "checkpoint.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "selection_manifest.json").write_text(json.dumps({"status": "selected", "best_valid_pearson": best}, indent=2) + "\n")
    return best
