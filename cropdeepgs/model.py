from __future__ import annotations

import numpy as np
import torch


class CropDeepGSNet(torch.nn.Module):
    """Crop genomic prediction network with optional environmental covariates."""

    def __init__(
        self,
        genotype_dim: int,
        env_dim: int,
        hidden: int = 448,
        dropout: float = 0.30,
        shortcut_scale: float = 0.03,
        interaction_scale: float = 0.10,
        head_depth: int = 2,
    ):
        super().__init__()
        self.env_dim = int(env_dim)
        self.shortcut_scale = float(shortcut_scale)
        self.interaction_scale = float(interaction_scale)
        self.genotype_encoder = torch.nn.Sequential(torch.nn.Linear(genotype_dim, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU())
        self.env_encoder = torch.nn.Sequential(torch.nn.Linear(max(env_dim, 1), hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU())
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden * 4, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Sigmoid(),
        )
        head_layers: list[torch.nn.Module] = [torch.nn.LayerNorm(hidden), torch.nn.Dropout(dropout)]
        for _ in range(max(1, int(head_depth))):
            head_layers.extend([torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)])
        head_layers.append(torch.nn.Linear(hidden, 1))
        self.head = torch.nn.Sequential(*head_layers)
        self.shortcut = torch.nn.Linear(genotype_dim + max(env_dim, 1), 1)

    def forward(self, genotype: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if env is None or self.env_dim == 0:
            env = torch.zeros((genotype.shape[0], 1), dtype=genotype.dtype, device=genotype.device)
        g = self.genotype_encoder(genotype)
        if self.env_dim == 0:
            z = g
        else:
            e = self.env_encoder(env)
            gate = self.gate(torch.cat([g, e, g * e, torch.abs(g - e)], dim=1))
            z = gate * g + (1.0 - gate) * e + self.interaction_scale * (g * e)
        pred = self.head(z).squeeze(-1)
        pred = pred + self.shortcut_scale * self.shortcut(torch.cat([genotype, env], dim=1)).squeeze(-1)
        return pred


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
