from __future__ import annotations
import torch
from torch import nn
from .adapters import FeatureAdapter, LabelAdapter

class ForecastMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = []
        width = input_size
        for _ in range(layers):
            blocks.extend([nn.Linear(width, hidden_size), nn.SiLU(), nn.Dropout(dropout)])
            width = hidden_size
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)

class DoubleAdaptModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, layers: int = 4,
                 dropout: float = 0.1, heads: int = 8, temperature: float = 10.0):
        super().__init__()
        self.feature_adapter = FeatureAdapter(input_size, heads, temperature)
        self.label_adapter = LabelAdapter(input_size, heads, temperature)
        self.base = ForecastMLP(input_size, hidden_size, layers, dropout)

    def forward(self, x: torch.Tensor, adapt_features: bool = True) -> torch.Tensor:
        return self.base(self.feature_adapter(x) if adapt_features else x)

    def adapted_label(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.label_adapter(x, y)
