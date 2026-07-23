from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .model import CropDeepGSNet, set_seed


def parse_cols(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, protocol: str, split: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    valid_correlation = len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if valid_correlation else float("nan")
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1]) if valid_correlation else float("nan")
    phenotype_sd = float(np.std(y_true, ddof=0))
    return {
        "model": "CropDeepGS",
        "protocol": protocol,
        "split": split,
        "n": int(len(y_true)),
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "nrmsep": rmse / phenotype_sd if phenotype_sd > 0 else float("nan"),
    }


def fit_genotype_transform(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test_scaled = scaler.transform(imputer.transform(x_test))
    components = min(int(n_components), x_train_scaled.shape[0] - 1, x_train_scaled.shape[1])
    if components < 1:
        return x_train_scaled.astype(np.float32), x_test_scaled.astype(np.float32)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    return (
        pca.fit_transform(x_train_scaled).astype(np.float32),
        pca.transform(x_test_scaled).astype(np.float32),
    )


def fit_environment_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.zeros((len(train), 0), dtype=np.float32), np.zeros((len(test), 0), dtype=np.float32)
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(train[column])]
    categorical = [column for column in columns if column not in numeric]
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    if numeric:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_numeric = imputer.fit_transform(train[numeric].astype(float))
        test_numeric = imputer.transform(test[numeric].astype(float))
        train_blocks.append(scaler.fit_transform(train_numeric).astype(np.float32))
        test_blocks.append(scaler.transform(test_numeric).astype(np.float32))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_values = train[categorical].fillna("missing").astype(str)
        test_values = test[categorical].fillna("missing").astype(str)
        train_blocks.append(encoder.fit_transform(train_values).astype(np.float32))
        test_blocks.append(encoder.transform(test_values).astype(np.float32))
    return np.hstack(train_blocks), np.hstack(test_blocks)


def _optimizer(
    model: CropDeepGSNet, args: argparse.Namespace, lr_scale: float = 1.0
) -> torch.optim.Optimizer:
    residual_ids = {id(parameter) for parameter in model.shortcut.parameters()}
    residual_ids.update({id(model.shortcut_log_gain), id(model.deep_log_gain), id(model.kernel_log_shrinkage)})
    residual_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in residual_ids and parameter.requires_grad
    ]
    nonlinear_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in residual_ids and parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {"params": nonlinear_parameters, "weight_decay": args.weight_decay},
            {"params": residual_parameters, "weight_decay": args.residual_weight_decay, "lr": args.residual_lr * lr_scale},
        ],
        lr=args.lr * lr_scale,
    )


def _training_loss(prediction: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    loss = (
        torch.nn.functional.mse_loss(prediction, target)
        if args.loss == "mse"
        else torch.nn.functional.smooth_l1_loss(prediction, target, beta=args.huber_beta)
    )
    if prediction.numel() >= 4 and args.correlation_loss_weight > 0:
        prediction_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        denominator = torch.sqrt(
            torch.sum(prediction_centered.square()) * torch.sum(target_centered.square()) + 1e-8
        )
        correlation = torch.sum(prediction_centered * target_centered) / denominator
        loss = loss + args.correlation_loss_weight * (1.0 - correlation)
    return loss


def train_cropdeepgs(
    xg_train: np.ndarray,
    xe_train: np.ndarray,
    y_train: np.ndarray,
    xg_test: np.ndarray,
    xe_test: np.ndarray,
    train_groups: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device(args.device)
    model = CropDeepGSNet(
        xg_train.shape[1],
        xe_train.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        shortcut_scale=args.residual_scale,
        interaction_scale=args.interaction_scale,
        head_depth=args.head_depth,
        spectral_mean=xg_train.mean(axis=0),
        spectral_variance=xg_train.var(axis=0),
        kernel_shrinkage=args.kernel_shrinkage,
        epistasis_rank=args.epistasis_rank,
        epistasis_scale=args.epistasis_scale,
        deep_scale=args.deep_scale,
        spectral_prefix_dim=xg_train.shape[1],
        additive_feature_start=0,
        additive_preserve_scale=False,
    ).to(device)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) or 1.0)
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    xg_tensor = torch.as_tensor(xg_train, dtype=torch.float32, device=device)
    xe_tensor = torch.as_tensor(xe_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
    indices = np.arange(len(y_train))
    validation_fraction = min(0.2, max(0.1, 200 / max(1, len(indices))))
    if len(np.unique(train_groups.astype(str))) >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        fit_indices, validation_indices = next(splitter.split(indices, groups=train_groups.astype(str)))
    else:
        fit_indices, validation_indices = train_test_split(
            indices, test_size=validation_fraction, random_state=seed
        )

    def fit_additive_layer(
        selected_indices: np.ndarray, residual_target: np.ndarray
    ) -> None:
        if args.analytic_shortcut_alpha <= 0:
            return
        model.eval()
        with torch.no_grad():
            environment = xe_tensor[selected_indices]
            if model.env_dim == 0 and environment.shape[1] == 0:
                environment = torch.zeros(
                    (len(selected_indices), 1), dtype=torch.float32, device=device
                )
            features = torch.cat(
                [model.additive_features(xg_tensor[selected_indices]), environment],
                dim=1,
            ).cpu().numpy()
        estimator = Ridge(
            alpha=float(args.analytic_shortcut_alpha),
            fit_intercept=True,
            solver="lsqr",
            tol=1e-6,
            max_iter=5000,
        )
        estimator.fit(features, residual_target)
        with torch.no_grad():
            gain = max(float(model.shortcut_gain().detach().cpu()), 1e-6)
            model.shortcut.weight.copy_(
                torch.as_tensor(
                    estimator.coef_[None, :] / gain,
                    dtype=model.shortcut.weight.dtype,
                    device=device,
                )
            )
            model.shortcut.bias.copy_(
                torch.as_tensor(
                    [float(estimator.intercept_) / gain],
                    dtype=model.shortcut.bias.dtype,
                    device=device,
                )
            )

    if args.analytic_shortcut_alpha > 0:
        fit_additive_layer(fit_indices, y_scaled[fit_indices])
        for parameter in model.shortcut.parameters():
            parameter.requires_grad_(False)
        model.shortcut_log_gain.requires_grad_(False)
        model.kernel_log_shrinkage.requires_grad_(False)

    if args.additive_warmup_epochs > 0:
        additive_parameters = list(model.shortcut.parameters()) + [
            model.shortcut_log_gain,
            model.kernel_log_shrinkage,
        ]
        warmup_optimizer = torch.optim.AdamW(
            additive_parameters,
            lr=args.residual_lr,
            weight_decay=args.residual_weight_decay,
        )
        model.train()
        warmup_rng = np.random.default_rng(seed + 104729)
        for _ in range(args.additive_warmup_epochs):
            order = warmup_rng.permutation(fit_indices)
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                prediction = model.shortcut_output(xg_tensor[batch], xe_tensor[batch])
                loss = torch.nn.functional.mse_loss(prediction, y_tensor[batch])
                warmup_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                warmup_optimizer.step()
    optimizer = _optimizer(model, args)
    rng = np.random.default_rng(seed)
    best_state = None
    best_correlation = -math.inf
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_order = rng.permutation(fit_indices)
        for start in range(0, len(fit_indices), args.batch_size):
            batch = epoch_order[start : start + args.batch_size]
            prediction = model(xg_tensor[batch], xe_tensor[batch])
            loss = _training_loss(prediction, y_tensor[batch], args)
            if args.kernel_alignment_weight > 0 and len(batch) >= 8:
                selected = batch[: min(len(batch), args.kernel_alignment_batch)]
                raw = torch.nn.functional.normalize(xg_tensor[selected], dim=1)
                latent = torch.nn.functional.normalize(model.genomic_latent(xg_tensor[selected]), dim=1)
                raw_kernel = raw @ raw.T
                latent_kernel = latent @ latent.T
                mask = ~torch.eye(len(selected), dtype=torch.bool, device=device)
                loss = loss + args.kernel_alignment_weight * torch.nn.functional.smooth_l1_loss(
                    latent_kernel[mask], raw_kernel[mask]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                xg_tensor[validation_indices], xe_tensor[validation_indices]
            ).cpu().numpy()
        validation_target = y_scaled[validation_indices]
        correlation = (
            float(np.corrcoef(validation_target, validation_prediction)[0, 1])
            if np.std(validation_target) > 0 and np.std(validation_prediction) > 0
            else -math.inf
        )
        if correlation > best_correlation + 1e-5:
            best_correlation = correlation
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.full_refit:
        optimizer = _optimizer(model, args, args.refit_lr_scale)
        model.train()
        for _ in range(args.refit_extra_epochs):
            order = rng.permutation(indices)
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                prediction = model(xg_tensor[batch], xe_tensor[batch])
                loss = _training_loss(prediction, y_tensor[batch], args)
                if args.kernel_alignment_weight > 0 and len(batch) >= 8:
                    selected = batch[: min(len(batch), args.kernel_alignment_batch)]
                    raw = torch.nn.functional.normalize(xg_tensor[selected], dim=1)
                    latent = torch.nn.functional.normalize(model.genomic_latent(xg_tensor[selected]), dim=1)
                    raw_kernel = raw @ raw.T
                    latent_kernel = latent @ latent.T
                    mask = ~torch.eye(len(selected), dtype=torch.bool, device=device)
                    loss = loss + args.kernel_alignment_weight * torch.nn.functional.smooth_l1_loss(
                        latent_kernel[mask], raw_kernel[mask]
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

    if args.analytic_shortcut_alpha > 0:
        model.eval()
        with torch.no_grad():
            nonlinear_prediction = (
                model(xg_tensor, xe_tensor)
                - model.shortcut_output(xg_tensor, xe_tensor)
            ).cpu().numpy()
        fit_additive_layer(indices, y_scaled - nonlinear_prediction)

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        xg_prediction = torch.as_tensor(xg_test, dtype=torch.float32, device=device)
        xe_prediction = torch.as_tensor(xe_test, dtype=torch.float32, device=device)
        inference_batch = args.batch_size * 8
        for start in range(0, len(xg_prediction), inference_batch):
            predictions.append(
                model(
                    xg_prediction[start : start + inference_batch],
                    xe_prediction[start : start + inference_batch],
                ).cpu().numpy()
            )
    return np.concatenate(predictions) * y_std + y_mean


def make_splits(
    frame: pd.DataFrame,
    sample_column: str,
    group_column: str | None,
    year_column: str | None,
    evaluations: list[str],
    folds: int,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    groups = frame[group_column or sample_column].astype(str).to_numpy()
    if "fivefold" in evaluations:
        if len(np.unique(groups)) >= folds:
            iterator = GroupKFold(n_splits=folds).split(np.arange(len(frame)), groups=groups)
        else:
            iterator = KFold(n_splits=min(folds, len(frame)), shuffle=True, random_state=47).split(frame)
        for fold, (train_indices, test_indices) in enumerate(iterator, start=1):
            splits.append(("fivefold_cv", f"fold{fold}", train_indices, test_indices))
    if "leave-year" in evaluations and year_column:
        year_values = frame[year_column].to_numpy()
        for year in sorted(frame[year_column].dropna().unique().tolist()):
            test_indices = np.where(year_values == year)[0]
            train_indices = np.where(year_values != year)[0]
            if len(train_indices) >= 80 and len(test_indices) >= 20:
                splits.append(("leave_year", f"year{year}", train_indices, test_indices))
    return splits


def run(args: argparse.Namespace) -> None:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    genotype = pd.read_csv(args.genotype, sep=None, engine="python")
    phenotype = pd.read_csv(args.phenotype, sep=None, engine="python")
    if args.sample_col not in genotype or args.sample_col not in phenotype:
        raise ValueError(f"Both input files must contain '{args.sample_col}'.")
    if args.trait not in phenotype:
        raise ValueError(f"Phenotype file does not contain trait '{args.trait}'.")
    marker_columns = [column for column in genotype if column != args.sample_col]
    data = phenotype.merge(genotype, on=args.sample_col, how="inner", validate="many_to_one")
    data = data.dropna(subset=[args.trait]).reset_index(drop=True)
    environment_columns = parse_cols(args.env_cols)
    missing_environment = [column for column in environment_columns if column not in data]
    if missing_environment:
        raise ValueError(f"Missing environmental columns: {missing_environment}")
    evaluations = parse_cols(args.eval)
    splits = make_splits(
        data, args.sample_col, args.group_col, args.year_col, evaluations, args.folds
    )
    if not splits:
        raise ValueError("No valid evaluation split was created.")

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for split_number, (protocol, split_name, train_indices, test_indices) in enumerate(splits):
        train = data.iloc[train_indices]
        test = data.iloc[test_indices]
        xg_train, xg_test = fit_genotype_transform(
            train[marker_columns].to_numpy(float),
            test[marker_columns].to_numpy(float),
            args.snp_pcs,
            args.seed + split_number,
        )
        xe_train, xe_test = fit_environment_transform(train, test, environment_columns)
        y_train = train[args.trait].to_numpy(float)
        y_test = test[args.trait].to_numpy(float)
        group_column = args.group_col or args.sample_col
        prediction = train_cropdeepgs(
            xg_train,
            xe_train,
            y_train,
            xg_test,
            xe_test,
            train[group_column].astype(str).to_numpy(),
            args,
            args.seed + 1009 * split_number,
        )
        metric_rows.append(metric_row(y_test, prediction, protocol, split_name))
        for sample_id, observed, predicted in zip(test[args.sample_col], y_test, prediction):
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "protocol": protocol,
                    "split": split_name,
                    "model": "CropDeepGS",
                    "y_true": observed,
                    "y_pred": predicted,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby(["model", "protocol"], as_index=False).agg(
        splits=("split", "nunique"),
        pearson=("pearson", "mean"),
        spearman=("spearman", "mean"),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
        nrmsep=("nrmsep", "mean"),
    )
    metrics.to_csv(output / "split_metrics.tsv", sep="\t", index=False)
    summary.to_csv(output / "summary_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.tsv", sep="\t", index=False)
    manifest = {
        "sample_column": args.sample_col,
        "trait": args.trait,
        "environment_columns": environment_columns,
        "marker_count": len(marker_columns),
        "record_count": len(data),
        "model": "CropDeepGS single end-to-end neural network",
        "parameters": {
            "snp_pcs": args.snp_pcs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "interaction_scale": args.interaction_scale,
            "head_depth": args.head_depth,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.lr,
            "correlation_loss_weight": args.correlation_loss_weight,
            "analytic_shortcut_alpha": args.analytic_shortcut_alpha,
            "seed": args.seed,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the single end-to-end CropDeepGS neural network."
    )
    parser.add_argument("--genotype", required=True, help="CSV/TSV table containing sample IDs and numeric SNP dosages.")
    parser.add_argument("--phenotype", required=True, help="CSV/TSV table containing sample IDs, the target trait and optional environmental variables.")
    parser.add_argument("--trait", required=True, help="Target phenotype column.")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--env-cols", default="", help="Comma-separated measured weather, soil or management variables.")
    parser.add_argument("--group-col", default=None, help="Genotype/group column kept intact between training and testing folds.")
    parser.add_argument("--year-col", default=None, help="Year column used only to define leave-one-year-out splits.")
    parser.add_argument("--eval", default="fivefold", help="Comma-separated evaluations: fivefold,leave-year.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--snp-pcs", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--deep-scale", type=float, default=1.0)
    parser.add_argument("--interaction-scale", type=float, default=0.25)
    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=152)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--residual-weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-lr", type=float, default=2e-3)
    parser.add_argument("--kernel-shrinkage", type=float, default=1e-3)
    parser.add_argument("--kernel-alignment-weight", type=float, default=0.0)
    parser.add_argument("--kernel-alignment-batch", type=int, default=128)
    parser.add_argument("--epistasis-rank", type=int, default=64)
    parser.add_argument("--epistasis-scale", type=float, default=0.10)
    parser.add_argument("--additive-warmup-epochs", type=int, default=0)
    parser.add_argument(
        "--analytic-shortcut-alpha",
        type=float,
        default=1000.0,
        help="Penalty for the in-network additive genomic output layer; use 0 to train that layer only by gradient descent.",
    )
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--loss", choices=["mse", "smoothl1"], default="mse")
    parser.add_argument("--correlation-loss-weight", type=float, default=0.05)
    parser.add_argument("--full-refit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refit-extra-epochs", type=int, default=8)
    parser.add_argument("--refit-lr-scale", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/cropdeepgs")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
