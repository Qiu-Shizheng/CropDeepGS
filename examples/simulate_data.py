#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    out = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    n = 160
    m = 300
    sample_id = np.array([f"line_{i:03d}" for i in range(n)])
    markers = rng.integers(0, 3, size=(n, m)).astype(float)
    years = rng.choice([2021, 2022, 2023, 2024], size=n)
    soil_n = rng.normal(0.0, 1.0, size=n)
    rain_mm = rng.normal(120.0, 22.0, size=n)
    irrigation = rng.choice(["low", "standard", "high"], size=n)
    causal = markers[:, :12] @ np.array([0.20, -0.15, 0.10, 0.08, -0.06, 0.05, 0.04, -0.04, 0.03, 0.03, -0.02, 0.02])
    irrigation_effect = pd.Series(irrigation).map({"low": -0.15, "standard": 0.0, "high": 0.18}).to_numpy()
    year_effect = (years - years.mean()) * 0.05
    env_effect = 0.18 * soil_n + 0.006 * (rain_mm - rain_mm.mean()) + irrigation_effect
    y = 5.0 + causal + env_effect + year_effect + rng.normal(0, 0.35, size=n)

    geno = pd.DataFrame(markers, columns=[f"snp_{j:04d}" for j in range(m)])
    geno.insert(0, "sample_id", sample_id)
    pheno = pd.DataFrame(
        {
            "sample_id": sample_id,
            "yield": y,
            "year": years,
            "soil_n": soil_n,
            "rain_mm": rain_mm,
            "irrigation": irrigation,
            "line_group": sample_id,
        }
    )
    geno.to_csv(out / "simulated_genotypes.tsv", sep="\t", index=False)
    pheno.to_csv(out / "simulated_phenotypes.tsv", sep="\t", index=False)
    print(out / "simulated_genotypes.tsv")
    print(out / "simulated_phenotypes.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
