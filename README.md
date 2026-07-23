# CropDeepGS

CropDeepGS is a Python toolkit for deep-learning genomic prediction in crops. It
accepts SNP dosage data, quantitative phenotypes and optional measured
environmental variables. The same command-line workflow can be used with any
crop whose inputs follow the documented table format.

[Chinese guide / 中文说明](README_zh.md)

## Features

- A PyTorch neural network with additive genomic, nonlinear genomic and
  genotype-by-environment components.
- Genotype-only prediction or joint genotype-environment prediction.
- Five-fold cross-validation with genetic groups kept in a single fold.
- Leave-one-year-out evaluation for multi-year data.
- Leakage-controlled imputation, scaling, encoding and SNP dimensionality
  reduction fitted within each training split.
- Per-split predictions, metrics, summary statistics and a complete run
  configuration.
- CPU and CUDA execution.

## Requirements

- Python 3.9 or later
- NumPy 1.23 or later
- pandas 1.5 or later
- scikit-learn 1.2 or later
- PyTorch 2.0 or later

The bundled example runs on a CPU. A CUDA GPU is recommended for larger
datasets.

## Installation

```bash
git clone https://github.com/Qiu-Shizheng/CropDeepGS.git
cd CropDeepGS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm the installation:

```bash
cropdeepgs --help
```

## Input files

CropDeepGS uses two comma- or tab-separated tables.

### Genotype table

The first column is normally `sample_id`. Every remaining column is a numeric
SNP dosage, usually coded as `0`, `1` or `2`. Missing dosages are allowed and
are imputed from the training samples in each validation split.

```text
sample_id  snp_0001  snp_0002  snp_0003
line_001   0         1         2
line_002   2         1         0
```

Each sample identifier must occur once in this table.

### Phenotype table

The phenotype table contains the same sample identifier, one target trait and
optional measured environmental variables. A sample may have several rows
when it was evaluated in multiple environments.

```text
sample_id  yield  year  soil_n  rain_mm  irrigation  line_group
line_001   5.42   2022  0.18    126.4    standard    line_001
line_002   4.91   2022  0.11    109.8    low         line_002
```

Environmental inputs may be numeric or categorical. Use measurements that are
available when predictions are made, such as soil properties, weather
summaries or management treatments. The target phenotype must not be included
among the environmental inputs.

## Quick start

Run the complete bundled example:

```bash
bash examples/run_example.sh
```

The script regenerates a small synthetic dataset and evaluates both supported
validation protocols.

Run CropDeepGS on another dataset:

```bash
cropdeepgs \
  --genotype path/to/genotypes.tsv \
  --phenotype path/to/phenotypes.tsv \
  --trait yield \
  --sample-col sample_id \
  --env-cols soil_n,rain_mm,irrigation \
  --group-col line_group \
  --year-col year \
  --eval fivefold,leave-year \
  --out results/yield
```

For genotype-only prediction, omit `--env-cols`:

```bash
cropdeepgs \
  --genotype path/to/genotypes.tsv \
  --phenotype path/to/phenotypes.tsv \
  --trait protein_content \
  --group-col line_group \
  --eval fivefold \
  --out results/protein_content
```

## Validation

### Five-fold cross-validation

`--eval fivefold` assigns each genetic group to one test fold and trains on the
other four folds. Set `--group-col` to a genotype, accession, line, family or
another biological grouping column. If it is omitted, `sample_id` is used.

### Leave-one-year-out

`--eval leave-year --year-col year` holds out one year at a time. SNP and
environmental preprocessing is fitted using the remaining years. A split is
created when at least 80 training records and 20 test records are available.
The year column defines the split and is not automatically used as a predictor.

Both protocols can be requested in one run with
`--eval fivefold,leave-year`.

## Outputs

The output directory contains:

| File | Contents |
|---|---|
| `predictions.tsv` | Observed and predicted phenotype for every test record |
| `split_metrics.tsv` | Pearson and Spearman correlations, RMSE, MAE, R-squared and NRMSEP for every split |
| `summary_metrics.tsv` | Mean metrics across splits in each validation protocol |
| `run_config.json` | Input columns, evaluation settings and model parameters |

Pearson and Spearman correlations and R-squared are higher when predictions are
better. RMSE, MAE and NRMSEP are lower when predictions are better. NRMSEP is
RMSE divided by the standard deviation of the observed phenotype in the test
split.

## Main options

| Option | Default | Description |
|---|---:|---|
| `--snp-pcs` | `512` | Maximum number of training-fitted SNP principal components |
| `--hidden` | `384` | Hidden width of the neural network |
| `--dropout` | `0.22` | Dropout probability |
| `--epochs` | `152` | Maximum training epochs |
| `--patience` | `18` | Early-stopping patience |
| `--batch-size` | `1024` | Training batch size |
| `--lr` | `0.00035` | Learning rate |
| `--seed` | `47` | Random seed |
| `--device` | automatic | `cuda`, `cuda:0` or `cpu` |

Use `cropdeepgs --help` for all advanced options.

## Computing resources

Memory use depends mainly on the number of records, SNPs and retained principal
components. Loading a dense SNP matrix requires approximately
`8 × samples × SNPs` bytes before preprocessing, plus pandas and scikit-learn
overhead. Convert very large datasets to a filtered numeric SNP matrix before
running the command. GPU memory is primarily controlled by `--batch-size`,
`--hidden` and `--snp-pcs`; reduce them in that order when CUDA memory is
insufficient.

## Reproducibility

The random seed is recorded in `run_config.json`. For strict comparison across
models, use identical folds, phenotype records and input variables. CUDA
kernels can still show small numerical differences across hardware and library
versions.

## Package layout

```text
cropdeepgs/                 Python package
examples/                   Synthetic input tables and runnable example
tests/                      Package smoke tests
README.md                   English user guide
README_zh.md                Chinese user guide
```

## License

CropDeepGS is released under the MIT License.
