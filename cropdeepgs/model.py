from __future__ import annotations

import numpy as np
import torch


class CropDeepGSNet(torch.nn.Module):
    """Genotype-context genomic prediction network.

    The model has four parts: a genotype encoder, a context encoder, a gated
    genotype-context interaction block and a small additive genomic shortcut.
    """

    def __init__(self, genotype_dim: int, context_dim: int, hidden: int = 192, dropout: float = 0.18, shortcut_scale: float = 0.15):
        super().__init__()
        self.context_dim = int(context_dim)
        self.shortcut_scale = float(shortcut_scale)
        self.genotype_encoder = torch.nn.Sequential(
            torch.nn.Linear(genotype_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
        )
        if context_dim > 0:
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(context_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.gate = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.Sigmoid())
        else:
            self.context_encoder = None
            self.gate = None
        self.shortcut = torch.nn.Linear(genotype_dim, 1)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, genotype: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        z = self.genotype_encoder(genotype)
        if self.context_encoder is not None and context is not None and context.shape[1] > 0:
            c = self.context_encoder(context)
            z = z * (1.0 + self.gate(c)) + c
        pred = self.head(z).squeeze(-1)
        pred = pred + self.shortcut_scale * self.shortcut(genotype).squeeze(-1)
        return pred


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
