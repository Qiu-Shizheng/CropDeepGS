# CropDeepGS

CropDeepGS is a unified deep learning framework for crop genomic prediction. It predicts quantitative crop traits from genome-wide marker data and, when available, measured environmental covariates such as soil, weather, irrigation, fertilizer or management variables.

This public repository is the user-facing software release. It does not include the large public datasets used in the manuscript, because those resources are hosted by their original repositories and may have separate download terms.

## Model Overview

CropDeepGS contains four components:

1. A genotype encoder for SNP dosage, marker, or other genome-wide numeric features.
2. An optional environmental covariate encoder for soil, weather, irrigation, fertilizer, or management variables.
3. A gated genotype-by-environment interaction block.
4. An additive genomic shortcut that keeps a stable linear genomic signal.

If no environmental covariate columns are supplied, CropDeepGS automatically runs as a genotype-only model.

## Installation

```bash
git clone https://github.com/Qiu-Shizheng/CropDeepGS.git
cd CropDeepGS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

GPU training is optional. Use `--device cpu` if CUDA is not available.

## Input Files

CropDeepGS accepts CSV or TSV files.

### Genotype Table

The genotype table must contain one sample identifier column and numeric marker columns.

| sample_id | snp_0001 | snp_0002 | snp_0003 |
|---|---:|---:|---:|
| line_001 | 0 | 1 | 2 |
| line_002 | 2 | 1 | 0 |

Markers may be SNP dosages coded as `0/1/2`, imputed marker values, or other numeric genome-wide features.

### Phenotype Table

The phenotype table must contain the same sample identifier column and one target trait column. Environmental covariate columns are optional.

| sample_id | yield | year | soil_n | rain_mm | irrigation | line_group |
|---|---:|---:|---:|---:|---|---|
| line_001 | 5.42 | 2023 | 0.12 | 141.3 | standard | line_001 |
| line_002 | 4.88 | 2024 | -0.36 | 109.7 | low | line_002 |

Use `--year-col` only for leave-one-year evaluation. The year column is used to define the held-out year and is not automatically used as an input feature. Only columns listed in `--env-cols` are used as environmental covariates.

Do not include phenotype-derived columns as model inputs. Examples to avoid include adjusted residuals from the target trait, target test labels, or any column calculated from the trait you want to predict.

## Quick Example

Run a small simulated example on CPU:

```bash
bash examples/run_example.sh
```

The example creates:

```text
examples/simulated_genotypes.tsv
examples/simulated_phenotypes.tsv
results/simulated_example/summary_metrics.tsv
results/simulated_example/split_metrics.tsv
results/simulated_example/predictions.tsv
results/simulated_example/run_config.json
```

## Train on Your Own Data

```bash
cropdeepgs \
  --genotype my_genotypes.tsv \
  --phenotype my_phenotypes.tsv \
  --trait grain_yield \
  --sample-col sample_id \
  --env-cols soil_n,rain_mm,irrigation \
  --group-col line_id \
  --year-col year \
  --eval fivefold,leave-year \
  --snp-pcs 128 \
  --epochs 80 \
  --device cuda \
  --out results/my_cropdeepgs_run
```

## Main Arguments

| Argument | Meaning |
|---|---|
| `--genotype` | Genotype table with marker columns. |
| `--phenotype` | Phenotype table with the target trait and optional environmental covariate columns. |
| `--trait` | Trait column to predict. |
| `--sample-col` | Shared sample identifier column. Default: `sample_id`. |
| `--env-cols` | Comma-separated environmental covariate columns. Leave empty for genotype-only prediction. |
| `--group-col` | Group used for five-fold validation. Use line, accession, hybrid, or another genetic-entry identifier when repeated records exist. |
| `--year-col` | Year column for leave-one-year evaluation. |
| `--eval` | `fivefold`, `leave-year`, or both separated by commas. |
| `--snp-pcs` | Number of genotype principal components. |
| `--device` | `cuda` or `cpu`. |
| `--baselines` | Optional baselines: `ridge,gblup`. Use an empty string to disable baselines. |
| `--out` | Output directory. |

## Output Files

| File | Contents |
|---|---|
| `summary_metrics.tsv` | Mean metrics by model and evaluation protocol. |
| `split_metrics.tsv` | Metrics for every fold or held-out year. |
| `predictions.tsv` | Observed and predicted trait values for each test record. |
| `run_config.json` | Input columns and model settings used in the run. |

Reported metrics include RMSE, MAE, Pearson correlation, R2 and NRMSEP. NRMSEP is RMSE divided by the standard deviation of the observed test-set phenotype, so lower values are better.

## Good Practice

Use grouped validation when the same genotype has multiple records. Use leave-one-year evaluation when you want to test temporal transfer. Provide environmental covariate columns only when they are true soil, weather or management measurements available before phenotyping.

## Citation

If you use CropDeepGS, please cite the CropDeepGS manuscript or preprint when available.
