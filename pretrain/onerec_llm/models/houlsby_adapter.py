import torch
from torch import nn


class HoulsbyAdapter(nn.Module):
    """The bottleneck residual adapter from Houlsby et al. (ICML 2019)."""

    def __init__(self, hidden_size: int, bottleneck_size: int, activation: str = "silu"):
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("hidden_size and bottleneck_size must be positive")
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.SiLU() if activation == "silu" else nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(hidden_states)))

