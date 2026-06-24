from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from .model import CropDeepGSNet, set_seed


def parse_cols(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, model: str, protocol: str, split: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0 else float("nan")
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    denom = float(np.std(y_true, ddof=0))
    return {
        "model": model,
        "protocol": protocol,
        "split": split,
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "r2": r2,
        "nrmsep": rmse / denom if denom > 0 else float("nan"),
    }


def fit_genotype_transform(x_train: np.ndarray, x_test: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)
    k = min(n_components, xtr.shape[0] - 1, xtr.shape[1])
    if k < 1:
        return xtr.astype(np.float32), xte.astype(np.float32)
    pca = PCA(n_components=k, random_state=0)
    return pca.fit_transform(xtr).astype(np.float32), pca.transform(xte).astype(np.float32)


def fit_environment_transform(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not cols:
        return np.zeros((len(train), 0), dtype=np.float32), np.zeros((len(test), 0), dtype=np.float32)
    train_env = train[cols].copy()
    test_env = test[cols].copy()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_env[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    if numeric_cols:
        scaler = StandardScaler()
        blocks_train.append(scaler.fit_transform(train_env[numeric_cols].astype(float)).astype(np.float32))
        blocks_test.append(scaler.transform(test_env[numeric_cols].astype(float)).astype(np.float32))
    for col in categorical_cols:
        levels = sorted(train_env[col].astype(str).fillna("NA").unique().tolist())
        mapping = {v: i for i, v in enumerate(levels)}
        tr = np.zeros((len(train_env), len(levels)), dtype=np.float32)
        te = np.zeros((len(test_env), len(levels)), dtype=np.float32)
        for i, val in enumerate(train_env[col].astype(str).fillna("NA")):
            tr[i, mapping[val]] = 1.0
        for i, val in enumerate(test_env[col].astype(str).fillna("NA")):
            if val in mapping:
                te[i, mapping[val]] = 1.0
        blocks_train.append(tr)
        blocks_test.append(te)
    return np.hstack(blocks_train).astype(np.float32), np.hstack(blocks_test).astype(np.float32)


def train_cropdeepgs(xg_train: np.ndarray, xe_train: np.ndarray, y_train: np.ndarray, xg_test: np.ndarray, xe_test: np.ndarray, args) -> np.ndarray:
    set_seed(args.seed)
    device = torch.device(args.device)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std <= 0:
        return np.full(len(xg_test), y_mean)
    y_scaled = (y_train - y_mean) / y_std
    model = CropDeepGSNet(
        xg_train.shape[1],
        xe_train.shape[1],
        args.hidden,
        args.dropout,
        args.shortcut_scale,
        args.interaction_scale,
        args.head_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xg = torch.tensor(xg_train, dtype=torch.float32, device=device)
    xe = torch.tensor(xe_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    n = len(y)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        order = rng.permutation(n)
        for start in range(0, n, args.batch_size):
            idx = torch.tensor(order[start : start + args.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(xg[idx], xe[idx]), y[idx], beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    preds = []
    with torch.no_grad():
        tg = torch.tensor(xg_test, dtype=torch.float32)
        te = torch.tensor(xe_test, dtype=torch.float32)
        for start in range(0, len(tg), args.batch_size * 8):
            preds.append(model(tg[start : start + args.batch_size * 8].to(device), te[start : start + args.batch_size * 8].to(device)).cpu().numpy())
    return np.concatenate(preds) * y_std + y_mean


def make_splits(df: pd.DataFrame, sample_col: str, group_col: str | None, year_col: str | None, evals: list[str], folds: int) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    splits = []
    n = len(df)
    groups = df[group_col].astype(str).to_numpy() if group_col else df[sample_col].astype(str).to_numpy()
    if "fivefold" in evals:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= folds:
            splitter = GroupKFold(n_splits=folds)
            iterator = splitter.split(np.arange(n), groups=groups)
        else:
            iterator = KFold(n_splits=min(folds, n), shuffle=True, random_state=1).split(np.arange(n))
        for i, (tr, te) in enumerate(iterator, start=1):
            splits.append(("fivefold", f"fold{i}", tr, te))
    if "leave-year" in evals and year_col:
        for year in sorted(df[year_col].dropna().unique().tolist()):
            te = np.where(df[year_col].to_numpy() == year)[0]
            tr = np.where(df[year_col].to_numpy() != year)[0]
            if len(tr) > 10 and len(te) > 0:
                splits.append(("leave-year", f"year{year}", tr, te))
    return splits


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geno = pd.read_csv(args.genotype, sep=None, engine="python")
    pheno = pd.read_csv(args.phenotype, sep=None, engine="python")
    if args.sample_col not in geno.columns or args.sample_col not in pheno.columns:
        raise ValueError(f"Both files must contain sample column '{args.sample_col}'.")
    if args.trait not in pheno.columns:
        raise ValueError(f"Phenotype file does not contain trait column '{args.trait}'.")
    marker_cols = [c for c in geno.columns if c != args.sample_col]
    data = pheno.merge(geno, on=args.sample_col, how="inner")
    data = data.dropna(subset=[args.trait]).reset_index(drop=True)
    env_cols = parse_cols(args.env_cols)
    missing_env = [c for c in env_cols if c not in data.columns]
    if missing_env:
        raise ValueError(f"Missing environmental covariate columns: {missing_env}")
    evals = parse_cols(args.eval)
    splits = make_splits(data, args.sample_col, args.group_col, args.year_col, evals, args.folds)
    if not splits:
        raise ValueError("No valid evaluation splits were created.")
    metrics = []
    predictions = []
    for protocol, split_name, train_idx, test_idx in splits:
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        xg_train, xg_test = fit_genotype_transform(train[marker_cols].to_numpy(float), test[marker_cols].to_numpy(float), args.snp_pcs)
        xe_train, xe_test = fit_environment_transform(train, test, env_cols)
        y_train = train[args.trait].to_numpy(float)
        y_test = test[args.trait].to_numpy(float)
        crop_pred = train_cropdeepgs(xg_train, xe_train, y_train, xg_test, xe_test, args)
        metrics.append(metric_row(y_test, crop_pred, "CropDeepGS", protocol, split_name))
        for sid, yt, yp in zip(test[args.sample_col], y_test, crop_pred):
            predictions.append({"sample_id": sid, "protocol": protocol, "split": split_name, "model": "CropDeepGS", "y_true": yt, "y_pred": yp})
        if args.baselines:
            if "ridge" in parse_cols(args.baselines):
                xtr = np.hstack([xg_train, xe_train])
                xte = np.hstack([xg_test, xe_test])
                pred = RidgeCV(alphas=np.logspace(-4, 5, 12)).fit(xtr, y_train).predict(xte)
                metrics.append(metric_row(y_test, pred, "Ridge", protocol, split_name))
            if "gblup" in parse_cols(args.baselines):
                pred = RidgeCV(alphas=np.logspace(-4, 5, 12)).fit(xg_train, y_train).predict(xg_test)
                metrics.append(metric_row(y_test, pred, "GBLUP", protocol, split_name))
    metrics_df = pd.DataFrame(metrics)
    summary = metrics_df.groupby(["model", "protocol"], as_index=False).agg(
        splits=("split", "nunique"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_pearson=("pearson", "mean"),
        mean_r2=("r2", "mean"),
        mean_nrmsep=("nrmsep", "mean"),
    )
    metrics_df.to_csv(out / "split_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out / "summary_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.tsv", sep="\t", index=False)
    manifest = {
        "genotype": str(args.genotype),
        "phenotype": str(args.phenotype),
        "sample_col": args.sample_col,
        "trait": args.trait,
        "environment_covariate_cols": env_cols,
        "marker_count": len(marker_cols),
        "records": len(data),
        "splits": len(splits),
        "model": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "shortcut_scale": args.shortcut_scale,
            "interaction_scale": args.interaction_scale,
            "head_depth": args.head_depth,
            "epochs": args.epochs,
        },
    }
    (out / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CropDeepGS on genotype and phenotype tables.")
    parser.add_argument("--genotype", required=True, help="CSV/TSV table with sample_id and numeric marker columns.")
    parser.add_argument("--phenotype", required=True, help="CSV/TSV table with sample_id, trait and optional environmental covariate columns.")
    parser.add_argument("--trait", required=True, help="Trait column in the phenotype table.")
    parser.add_argument("--sample-col", default="sample_id", help="Shared sample identifier column.")
    parser.add_argument("--env-cols", default="", help="Comma-separated environmental covariate columns, e.g. soil_n,rain_mm,irrigation.")
    parser.add_argument("--group-col", default=None, help="Column used to keep related records together in five-fold validation. Defaults to sample column.")
    parser.add_argument("--year-col", default=None, help="Year column for leave-year validation.")
    parser.add_argument("--eval", default="fivefold", help="Comma-separated protocols: fivefold,leave-year.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--snp-pcs", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=448)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--shortcut-scale", type=float, default=0.03)
    parser.add_argument("--interaction-scale", type=float, default=0.10)
    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baselines", default="ridge,gblup", help="Optional baselines: ridge,gblup. Empty string disables baselines.")
    parser.add_argument("--out", default="results/cropdeepgs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
