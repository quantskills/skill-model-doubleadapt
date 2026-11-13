from __future__ import annotations
import math
import torch
from torch import nn

def cosine_gate(x: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    return torch.softmax((x @ prototypes.T) / temperature, dim=-1)

class FeatureAdapter(nn.Module):
    def __init__(self, input_size: int, heads: int = 8, temperature: float = 10.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(heads, input_size))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
        self.transforms = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(heads)])
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = cosine_gate(x, self.prototypes, self.temperature)
        adapted = torch.zeros_like(x)
        for index, transform in enumerate(self.transforms):
            adapted = adapted + weights[:, index:index + 1] * transform(x)
        return x + adapted

class LabelAdapter(nn.Module):
    def __init__(self, input_size: int, heads: int = 8, temperature: float = 10.0, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.prototypes = nn.Parameter(torch.empty(heads, hidden_size))
        self.scale = nn.Parameter(torch.empty(1, heads))
        self.bias = nn.Parameter(torch.zeros(1, heads))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
        nn.init.uniform_(self.scale, 0.75, 1.25)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = cosine_gate(self.encoder(x), self.prototypes, self.temperature)
        transformed = self.scale * y.reshape(-1, 1) + self.bias
        return (weights * transformed).sum(dim=-1)
