# Federated physics-informed graph neural networks for site-agnostic prediction of lipid nanoparticle organ tropism

Fed-PI-GNN predicts lipid nanoparticle delivery across organ compartments with a molecular graph encoder, a Stokes–Einstein transport kernel, a PBPK mass-continuity residual, a Gibbs protonation constraint and federated aggregation with client-specific classification heads. Six biodistribution sources remain separate clients throughout training.

## Installation

Python 3.11 and CUDA 12.4 are supported.

```bash
pip install -e .
```

Conda users can run `conda env create -f environment.yml`. A container can be built with `docker build -t fed-pi-gnn .`.

## Data

Authoritative dataset pages, versions and licences are listed in `dataset_links.txt`. Prepare one local directory per client. Each record must provide a molecular graph, formulation fractions, hydrodynamic radius, apparent pKa, organ regression targets and top-quartile classification targets. Raw records from different licences must not be combined into one file.

Expected client counts are 258 Paunovska formulations, 385 Lokugamage formulations, 96 SORT formulations, 1,200 AGILE formulations, 1,100 LANTERN formulations and 520 Witten formulations. Split manifests use canonical-SMILES SHA-256 values and must have zero train-test overlap.

## Training

```bash
fed-pi-train --config configs/main.yaml --output runs/main.pt
```

The main setting uses all six clients for 50 communication rounds, five local epochs per round, batch size 128 per client and 20 independent seeds. The reported full experimental matrix consumed about 19,200 A100-80GB GPU-hours. A single default run was measured on eight A100-40GB GPUs at 10.8 GB per GPU and about 8.4 hours.

## Evaluation

```bash
fed-pi-evaluate --config configs/main.yaml --output runs/main.pt
```

The paper reports LANTERN random-split R² 0.851 ± 0.008, Murcko-scaffold R² 0.563 ± 0.021 and AUROC 0.841 ± 0.012 for liver, 0.827 ± 0.014 for spleen, 0.813 ± 0.018 for lung and 0.788 ± 0.022 for bone marrow across 20 seeds. Evaluation includes MSE, Brier score, expected calibration error, paired bootstrap intervals, Holm–Bonferroni correction, leave-one-database-out transfer and nine-stratum worst-group disparity.

## Project layout

`code/fed_pi_gnn` contains configuration, graph data, physics operators, the GAT model, objectives, federation, training, evaluation and command-line entry points. `configs/main.yaml` records the paper setting. `configs/smoke.yaml` is reserved for the automated training test. `tests` covers tensor operators, metrics, aggregation, batching and the end-to-end training path.
