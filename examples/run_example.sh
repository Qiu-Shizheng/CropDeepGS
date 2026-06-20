#!/usr/bin/env bash
set -euo pipefail

python examples/simulate_data.py

cropdeepgs \
  --genotype examples/simulated_genotypes.tsv \
  --phenotype examples/simulated_phenotypes.tsv \
  --trait yield \
  --sample-col sample_id \
  --context-cols location,maturity_group \
  --group-col line_group \
  --year-col year \
  --eval fivefold,leave-year \
  --snp-pcs 64 \
  --epochs 30 \
  --device cpu \
  --out results/simulated_example
