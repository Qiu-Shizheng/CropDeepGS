# CropDeepGS

CropDeepGS is a unified deep learning framework for crop genomic prediction. It predicts quantitative crop traits from genome-wide marker data and optional prediction-time context, such as location, maturity group, soil, weather or management variables.

The public release is intended for users who want to train CropDeepGS on their own crop genotype and phenotype tables. It does not include raw public datasets from the manuscript because many public breeding datasets have their own repository terms and large file sizes.

## What the Model Does

CropDeepGS uses four components:

1. A genotype encoder for SNP or marker features.
2. A context encoder for variables known before phenotyping.
3. A gated genotype-by-context interaction block.
4. An additive genomic shortcut that keeps a stable GBLUP-like signal.

The command line tool reports RMSE, MAE, Pearson correlation, R2 and NRMSEP. NRMSEP is RMSE divided by the standard deviation of the observed test-set phenotype, so lower values are better.

## Installation

Create a clean Python environment, then install the package:

```bash
git clone https://github.com/Qiu-Shizheng/CropDeepGS.git
cd CropDeepGS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

GPU training is optional. If your local PyTorch installation does not support CUDA, use `--device cpu`.

## Input Files

CropDeepGS expects two tabular files. CSV and TSV are both accepted.

### Genotype Table

The genotype file must contain one sample identifier column and numeric marker columns:

| sample_id | snp_0001 | snp_0002 | snp_0003 |
|---|---:|---:|---:|
| line_001 | 0 | 1 | 2 |
| line_002 | 2 | 1 | 0 |

Markers can be SNP dosages coded as `0/1/2`, imputed marker values, or other numeric genome-wide features. The software standardizes markers and reduces them to genotype principal components inside each training split.

### Phenotype Table

The phenotype file must contain the same sample identifier column, one trait column and optional context columns:

| sample_id | yield | location | year | maturity_group |
|---|---:|---|---:|---|
| line_001 | 5.42 | LOC_A | 2023 | MG2 |
| line_002 | 4.88 | LOC_B | 2023 | MG1 |

Context columns should only include information that is known before phenotyping. Do not include sample IDs, trial IDs, historical trait means, adjusted residuals, target test labels or any column derived from the phenotype you are trying to predict.

## Quick Example

Generate a small simulated dataset and run CropDeepGS on CPU:

```bash
bash examples/run_example.sh
```

The command creates:

```text
examples/simulated_genotypes.tsv
examples/simulated_phenotypes.tsv
results/simulated_example/summary_metrics.tsv
results/simulated_example/split_metrics.tsv
results/simulated_example/predictions.tsv
results/simulated_example/run_config.json
```

## Train on Your Own Data

Example command:

```bash
cropdeepgs \
  --genotype my_genotypes.tsv \
  --phenotype my_phenotypes.tsv \
  --trait grain_yield \
  --sample-col sample_id \
  --context-cols location,maturity_group \
  --group-col line_id \
  --year-col year \
  --eval fivefold,leave-year \
  --snp-pcs 128 \
  --epochs 80 \
  --device cuda \
  --out results/my_cropdeepgs_run
```

Arguments:

| Argument | Meaning |
|---|---|
| `--genotype` | Genotype table with marker columns. |
| `--phenotype` | Phenotype table with the target trait and optional context columns. |
| `--trait` | Trait column to predict. |
| `--sample-col` | Shared sample identifier column. Default: `sample_id`. |
| `--context-cols` | Comma-separated context columns. Leave empty if no context is available. |
| `--group-col` | Group used for five-fold validation. Use line/accession/hybrid ID to avoid leakage. |
| `--year-col` | Year column for leave-one-year validation. |
| `--eval` | `fivefold`, `leave-year`, or both. |
| `--snp-pcs` | Number of genotype principal components. |
| `--device` | `cuda` or `cpu`. |
| `--baselines` | Optional baselines: `ridge,gblup`. Use an empty string to disable baselines. |
| `--out` | Output directory. |

## Output Files

`summary_metrics.tsv` reports average metrics by model and protocol. `split_metrics.tsv` reports metrics for every fold or held-out year. `predictions.tsv` contains observed and predicted trait values for each test record. `run_config.json` records the input columns and model settings used in the run.

## Good Practice

Use grouped validation whenever the same genotype has multiple records. Use leave-one-year validation when testing temporal transfer. Do not provide trial identifiers or phenotype-derived covariates as context features. If leave-one-year validation is used, avoid using the held-out year itself as a model feature unless the year is truly a known deployment covariate in your breeding program.

## Citation

If you use CropDeepGS, please cite the CropDeepGS manuscript or preprint when available. The software release implements the public user-facing training interface for the unified CropDeepGS framework.
