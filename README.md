# CausalTwin-Edu

This repository contains the scripts used for the CausalTwin-Edu experiments. The code works with local copies of the public datasets; no data are downloaded by the scripts.

## Files

```text
project_root/
  causaltwin_edu_reproducible.py
  intervention_ablation_reproducible.py
  make_manuscript_figures.py
  requirements.txt
  requirements_optional.txt
  datasets/
    README.md
```

Raw datasets and generated results are not included.

## Data

Two public datasets are used:

1. Open University Learning Analytics Dataset (OULAD)
2. UCI Predict Students' Dropout and Academic Success dataset

Download the datasets from their official sources and place the files as shown below.

### OULAD

Put the OULAD CSV files here:

```text
datasets/raw/oulad/
```

Required files:

```text
studentInfo.csv
studentVle.csv
studentAssessment.csv
assessments.csv
vle.csv
```

The OULAD files are not redistributed in this repository, so please keep your local copy under the path above.

### UCI dropout and academic success dataset

Put the UCI CSV file here:

```text
datasets/raw/uci_dropout_success/
```

The file must contain the original `Target` column. A typical local path is:

```text
datasets/raw/uci_dropout_success/data.csv
```

For the external validation task, the `Enrolled` class is removed and the binary task uses only `Dropout` and `Graduate`.

## Installation

Install the main dependencies:

```bash
pip install -r requirements.txt
```

LightGBM is optional:

```bash
pip install -r requirements_optional.txt
```

If LightGBM is not available, the script skips that baseline and continues with the scikit-learn models.

## Run

Main experiment:

```bash
python causaltwin_edu_reproducible.py
```

Intervention ablation study:

```bash
python intervention_ablation_reproducible.py
```

Regenerate manuscript figures from saved CSV results:

```bash
python make_manuscript_figures.py
```

The main scripts write results to:

```text
results_applied_intelligence/
  csv/
  figures/
  reports/
```

Clean figure files are written to:

```text
submission_figures_clean/
  main_manuscript_figures/
  optional_extra_figures/
```

## Reproducibility notes

Random seeds are fixed where they are used. Small numerical differences may still appear across Python versions, operating systems, or package versions.

The intervention outputs are model-based counterfactual recommendations. They are meant for decision-support analysis and should not be interpreted as confirmed causal effects from real classroom interventions.
