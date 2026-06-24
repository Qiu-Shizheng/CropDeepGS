#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
CROPDEEPGS_BIN="${CROPDEEPGS:-cropdeepgs}"

"$PYTHON_BIN" examples/simulate_data.py

"$CROPDEEPGS_BIN" \
  --genotype examples/simulated_genotypes.tsv \
  --phenotype examples/simulated_phenotypes.tsv \
  --trait yield \
  --sample-col sample_id \
  --env-cols soil_n,rain_mm,irrigation \
  --group-col line_group \
  --year-col year \
  --eval fivefold,leave-year \
  --snp-pcs 64 \
  --epochs 30 \
  --device cpu \
  --out results/simulated_example
