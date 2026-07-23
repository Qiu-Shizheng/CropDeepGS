from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cropdeepgs import cli
from cropdeepgs.model import CropDeepGSNet


def test_model_handles_genotype_only_and_environment_inputs():
    for genotype_dim, environment_dim in [(64, 0), (64, 5)]:
        model = CropDeepGSNet(genotype_dim, environment_dim)
        prediction = model(
            torch.randn(7, genotype_dim),
            torch.randn(7, environment_dim),
        )
        assert prediction.shape == (7,)
        assert torch.isfinite(prediction).all()


def test_genotype_only_additive_path_has_consistent_shape():
    model = CropDeepGSNet(64, 0)
    prediction = model.shortcut_output(torch.randn(7, 64), torch.zeros(7, 0))
    assert prediction.shape == (7,)
    assert torch.isfinite(prediction).all()


def test_cli_defaults_are_stable():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--genotype", "genotypes.tsv", "--phenotype", "phenotypes.tsv", "--trait", "yield"]
    )
    assert args.snp_pcs == 512
    assert args.hidden == 384
    assert np.isclose(args.dropout, 0.22)
    assert np.isclose(args.interaction_scale, 0.25)
    assert np.isclose(args.correlation_loss_weight, 0.05)
    assert args.loss == "mse"
    assert np.isclose(args.residual_scale, 0.20)
    assert np.isclose(args.deep_scale, 1.0)
    assert np.isclose(args.kernel_shrinkage, 1e-3)
    assert np.isclose(args.kernel_alignment_weight, 0.0)


def test_grouped_folds_do_not_split_groups():
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(20)],
            "family": [f"family_{i // 2}" for i in range(20)],
        }
    )
    splits = cli.make_splits(
        frame,
        sample_column="sample_id",
        group_column="family",
        year_column=None,
        evaluations=["fivefold"],
        folds=5,
    )
    for _, _, train_indices, test_indices in splits:
        train_groups = set(frame.iloc[train_indices]["family"])
        test_groups = set(frame.iloc[test_indices]["family"])
        assert train_groups.isdisjoint(test_groups)


def test_marker_and_spectral_inputs_use_distinct_kernel_normalization():
    mean = np.zeros(2, dtype=np.float32)
    variance = np.array([1.0, 4.0], dtype=np.float32)
    marker_model = CropDeepGSNet(
        2, 0, spectral_mean=mean, spectral_variance=variance, kernel_shrinkage=1e-4
    )
    spectral_model = CropDeepGSNet(
        2,
        0,
        spectral_mean=mean,
        spectral_variance=variance,
        kernel_shrinkage=1e-4,
        spectral_prefix_dim=2,
    )
    genotype = torch.tensor([[1.0, 2.0]])
    marker_features = marker_model.spectral_features(genotype)
    spectral_features = spectral_model.spectral_features(genotype)
    assert torch.allclose(marker_features, torch.ones_like(marker_features), atol=1e-3)
    assert spectral_features[0, 1] > marker_features[0, 1] * 1.9


if __name__ == "__main__":
    test_model_handles_genotype_only_and_environment_inputs()
    test_genotype_only_additive_path_has_consistent_shape()
    test_cli_defaults_are_stable()
    test_grouped_folds_do_not_split_groups()
    test_marker_and_spectral_inputs_use_distinct_kernel_normalization()
    print("5 package tests passed")
