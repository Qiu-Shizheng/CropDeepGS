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
    locations = rng.choice(["LOC_A", "LOC_B", "LOC_C"], size=n)
    years = rng.choice([2021, 2022, 2023, 2024], size=n)
    maturity_group = rng.choice(["MG1", "MG2", "MG3"], size=n)
    causal = markers[:, :12] @ np.array([0.20, -0.15, 0.10, 0.08, -0.06, 0.05, 0.04, -0.04, 0.03, 0.03, -0.02, 0.02])
    loc_effect = pd.Series(locations).map({"LOC_A": 0.3, "LOC_B": -0.1, "LOC_C": 0.0}).to_numpy()
    year_effect = (years - years.mean()) * 0.08
    mg_effect = pd.Series(maturity_group).map({"MG1": -0.2, "MG2": 0.0, "MG3": 0.25}).to_numpy()
    y = 5.0 + causal + loc_effect + year_effect + mg_effect + rng.normal(0, 0.35, size=n)

    geno = pd.DataFrame(markers, columns=[f"snp_{j:04d}" for j in range(m)])
    geno.insert(0, "sample_id", sample_id)
    pheno = pd.DataFrame(
        {
            "sample_id": sample_id,
            "yield": y,
            "location": locations,
            "year": years,
            "maturity_group": maturity_group,
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
