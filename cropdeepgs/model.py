from __future__ import annotations

import numpy as np
import torch


class CropDeepGSNet(torch.nn.Module):
    """Crop genomic prediction network with optional environmental descriptors.

    The model has four parts: a genotype encoder, an environmental descriptor
    encoder, a gated genotype-by-environment block and a small additive genomic
    shortcut.
    """

    def __init__(self, genotype_dim: int, env_dim: int, hidden: int = 192, dropout: float = 0.18, shortcut_scale: float = 0.15):
        super().__init__()
        self.env_dim = int(env_dim)
        self.shortcut_scale = float(shortcut_scale)
        self.genotype_encoder = torch.nn.Sequential(
            torch.nn.Linear(genotype_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
        )
        if env_dim > 0:
            self.env_encoder = torch.nn.Sequential(
                torch.nn.Linear(env_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.gate = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.Sigmoid())
        else:
            self.env_encoder = None
            self.gate = None
        self.shortcut = torch.nn.Linear(genotype_dim, 1)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, genotype: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        z = self.genotype_encoder(genotype)
        if self.env_encoder is not None and env is not None and env.shape[1] > 0:
            e = self.env_encoder(env)
            z = z * (1.0 + self.gate(e)) + e
        pred = self.head(z).squeeze(-1)
        pred = pred + self.shortcut_scale * self.shortcut(genotype).squeeze(-1)
        return pred


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
