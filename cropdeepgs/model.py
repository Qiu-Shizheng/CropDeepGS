from __future__ import annotations

import math

import numpy as np
import torch


class CropDeepGSNet(torch.nn.Module):
    """Neural genomic prediction model with optional environmental inputs."""

    def __init__(
        self,
        genotype_dim: int,
        env_dim: int,
        hidden: int = 384,
        dropout: float = 0.22,
        shortcut_scale: float = 0.20,
        interaction_scale: float = 0.25,
        head_depth: int = 2,
        shortcut_mode: str = "learnable",
        shortcut_norm: str = "none",
        deep_scale: float = 1.0,
        deep_mode: str = "learnable",
        spectral_mean: np.ndarray | None = None,
        spectral_variance: np.ndarray | None = None,
        kernel_shrinkage: float = 1e-3,
        epistasis_rank: int = 64,
        epistasis_scale: float = 0.10,
        spectral_prefix_dim: int = 0,
        additive_feature_start: int = 0,
        additive_preserve_scale: bool = False,
    ):
        super().__init__()
        self.env_dim = int(env_dim)
        self.shortcut_mode = str(shortcut_mode)
        self.deep_mode = str(deep_mode)
        self.interaction_scale = float(interaction_scale)
        self.epistasis_scale = float(epistasis_scale)
        self.spectral_prefix_dim = max(
            0, min(int(spectral_prefix_dim), int(genotype_dim))
        )
        requested_start = max(0, min(int(additive_feature_start), int(genotype_dim)))
        self.additive_feature_start = (
            0 if requested_start >= int(genotype_dim) else requested_start
        )
        self.additive_preserve_scale = bool(additive_preserve_scale)
        mean = np.zeros(genotype_dim, dtype=np.float32) if spectral_mean is None else np.asarray(spectral_mean, dtype=np.float32)
        variance = np.ones(genotype_dim, dtype=np.float32) if spectral_variance is None else np.asarray(spectral_variance, dtype=np.float32)
        if mean.shape != (genotype_dim,) or variance.shape != (genotype_dim,):
            raise ValueError("Spectral mean and variance must match genotype_dim")
        self.register_buffer("spectral_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("spectral_variance", torch.tensor(np.maximum(variance, 1e-7), dtype=torch.float32))
        shrinkage = max(float(kernel_shrinkage), 1e-4)
        self.kernel_log_shrinkage = torch.nn.Parameter(
            torch.tensor(math.log(math.expm1(shrinkage)), dtype=torch.float32)
        )
        self.genotype_encoder = torch.nn.Sequential(torch.nn.Linear(genotype_dim, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU())
        self.env_encoder = torch.nn.Sequential(torch.nn.Linear(max(env_dim, 1), hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU())
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden * 4, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Sigmoid(),
        )
        rank = max(8, min(int(epistasis_rank), hidden))
        self.epistasis_left = torch.nn.Linear(hidden, rank, bias=False)
        self.epistasis_right = torch.nn.Linear(hidden, rank, bias=False)
        self.epistasis_projection = torch.nn.Linear(rank, hidden, bias=False)
        self.environment_film = (
            torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden * 2),
            )
            if self.env_dim > 0
            else None
        )
        head_layers: list[torch.nn.Module] = [torch.nn.LayerNorm(hidden), torch.nn.Dropout(dropout)]
        for _ in range(max(1, int(head_depth))):
            head_layers.extend([torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout)])
        head_layers.append(torch.nn.Linear(hidden, 1))
        self.head = torch.nn.Sequential(*head_layers)
        shortcut_dim = (genotype_dim - self.additive_feature_start) + max(env_dim, 1)
        if shortcut_norm == "batch":
            self.shortcut_norm = torch.nn.BatchNorm1d(shortcut_dim, affine=False)
        elif shortcut_norm == "layer":
            self.shortcut_norm = torch.nn.LayerNorm(shortcut_dim, elementwise_affine=False)
        elif shortcut_norm == "none":
            self.shortcut_norm = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown shortcut_norm: {shortcut_norm}")
        self.shortcut = torch.nn.Linear(shortcut_dim, 1)
        torch.nn.init.zeros_(self.shortcut.weight)
        torch.nn.init.zeros_(self.shortcut.bias)
        torch.nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.head[-1].bias)
        if self.shortcut_mode == "learnable":
            init = max(float(shortcut_scale), 1e-4)
            self.shortcut_log_gain = torch.nn.Parameter(torch.tensor(math.log(math.expm1(init)), dtype=torch.float32))
        elif self.shortcut_mode == "fixed":
            self.register_buffer("shortcut_log_gain", torch.tensor(max(float(shortcut_scale), 0.0), dtype=torch.float32))
        else:
            raise ValueError(f"Unknown shortcut_mode: {self.shortcut_mode}")
        deep_init = max(float(deep_scale), 0.0)
        if self.deep_mode == "learnable":
            self.deep_log_gain = torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(deep_init, 1e-4))), dtype=torch.float32))
        elif self.deep_mode == "fixed":
            self.register_buffer("deep_log_gain", torch.tensor(float(deep_init), dtype=torch.float32))
        else:
            raise ValueError(f"Unknown deep_mode: {self.deep_mode}")

    def kernel_shrinkage(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.kernel_log_shrinkage)

    def spectral_features(self, genotype: torch.Tensor) -> torch.Tensor:
        centered = genotype - self.spectral_mean
        pieces: list[torch.Tensor] = []
        if self.spectral_prefix_dim:
            variance = self.spectral_variance[: self.spectral_prefix_dim]
            positive = variance[variance > 1e-7]
            reference = torch.median(positive) if positive.numel() else variance.new_tensor(1.0)
            shrinkage = torch.sqrt(
                variance / (variance + self.kernel_shrinkage() * reference)
            )
            pieces.append(
                centered[:, : self.spectral_prefix_dim] * shrinkage / torch.sqrt(reference)
            )
        if self.spectral_prefix_dim < centered.shape[1]:
            variance = self.spectral_variance[self.spectral_prefix_dim :]
            positive = variance[variance > 1e-7]
            reference = torch.median(positive) if positive.numel() else variance.new_tensor(1.0)
            pieces.append(
                centered[:, self.spectral_prefix_dim :]
                / torch.sqrt(variance + self.kernel_shrinkage() * reference)
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    def genomic_latent(self, genotype: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral_features(genotype)
        g = self.genotype_encoder(spectral)
        interaction = self.epistasis_left(g) * self.epistasis_right(g)
        return g + self.epistasis_scale * self.epistasis_projection(interaction)

    def shortcut_gain(self) -> torch.Tensor:
        if self.shortcut_mode == "fixed":
            return self.shortcut_log_gain
        return torch.nn.functional.softplus(self.shortcut_log_gain)

    def deep_gain(self) -> torch.Tensor:
        if self.deep_mode == "fixed":
            return self.deep_log_gain
        return torch.nn.functional.softplus(self.deep_log_gain)

    def shortcut_output(self, genotype: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        if self.env_dim == 0 and env.shape[1] == 0:
            env = torch.zeros((genotype.shape[0], 1), dtype=genotype.dtype, device=genotype.device)
        x = torch.cat([self.additive_features(genotype), env], dim=1)
        return self.shortcut_gain() * self.shortcut(self.shortcut_norm(x)).squeeze(-1)

    def additive_features(self, genotype: torch.Tensor) -> torch.Tensor:
        if self.additive_preserve_scale:
            return (genotype - self.spectral_mean)[:, self.additive_feature_start :]
        return self.spectral_features(genotype)[:, self.additive_feature_start :]

    def forward(self, genotype: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if env is None or self.env_dim == 0:
            env = torch.zeros((genotype.shape[0], 1), dtype=genotype.dtype, device=genotype.device)
        g = self.genomic_latent(genotype)
        if self.env_dim == 0:
            z = g
        else:
            e = self.env_encoder(env)
            gamma, beta = self.environment_film(e).chunk(2, dim=1)
            g = g * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            gate = self.gate(torch.cat([g, e, g * e, torch.abs(g - e)], dim=1))
            z = gate * g + (1.0 - gate) * e + self.interaction_scale * (g * e)
        pred = self.deep_gain() * self.head(z).squeeze(-1)
        pred = pred + self.shortcut_output(genotype, env)
        return pred


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
