<div align="center">

# [CNN-LSTM-DDoS-Detection-CICDDoS2019.](https://github.com/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019) <img src="https://cdn.simpleicons.org/github" width="3%" height="3%">

</div>

<div align="center">

---

**Methodology source:** Deepak Singh Rajput and Arvind Kumar Upadhyay, *Enhanced Network Defense: Optimized Multi-Layer Ensemble for DDoS Attack Detection* (2024), DOI: [10.52756/ijerr.2024.v46.020](https://doi.org/10.52756/ijerr.2024.v46.020).

A reproducible, memory-aware implementation of the paper's 12-class CICDDoS2019 CNN-LSTM experiment, organized as a modular Python project for Apple Silicon and Linux GPU servers. This repository is an independent reproduction and is not an official repository of the paper's authors.

---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Contributors](https://img.shields.io/github/contributors/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![GitHub Created At](https://img.shields.io/github/created-at/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/c2839086290a2fe2e4276baf5c525323c1fe7b8f.svg "Repobeats analytics image")

</div>

## Table of Contents

- [CNN-LSTM-DDoS-Detection-CICDDoS2019. ](#cnn-lstm-ddos-detection-cicddos2019-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Original Paper](#original-paper)
  - [Reproduction Objective](#reproduction-objective)
  - [Paper Methodology and Project Mapping](#paper-methodology-and-project-mapping)
  - [Pipeline](#pipeline)
  - [Target Classes](#target-classes)
  - [CNN-LSTM Architecture](#cnn-lstm-architecture)
  - [Dataset Handling and Sampling](#dataset-handling-and-sampling)
    - [Read-only source protection](#read-only-source-protection)
    - [Bounded Mac mode](#bounded-mac-mode)
    - [Full Linux mode](#full-linux-mode)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [Setup](#setup)
    - [Clone the repository](#clone-the-repository)
  - [Makefile Automation](#makefile-automation)
  - [Installation and Execution](#installation-and-execution)
    - [macOS Apple Silicon — bounded-memory execution](#macos-apple-silicon--bounded-memory-execution)
    - [Linux SSH server — full-source execution](#linux-ssh-server--full-source-execution)
  - [Runtime Logging and Completion Sound](#runtime-logging-and-completion-sound)
  - [Configuration](#configuration)
  - [Generated Outputs](#generated-outputs)
  - [Reproducibility Notes and Limitations](#reproducibility-notes-and-limitations)
  - [Results Target](#results-target)
  - [References](#references)
  - [How to Cite](#how-to-cite)
  - [Contributing](#contributing)
  - [Author](#author)
  - [License](#license)
    - [MIT License](#mit-license)

## Introduction

This repository reconstructs the **12-class CICDDoS2019 CNN-LSTM experiment** described by Deepak Singh Rajput and Arvind Kumar Upadhyay in *Enhanced Network Defense: Optimized Multi-Layer Ensemble for DDoS Attack Detection*.

The paper compares a hybrid machine-learning approach with a CNN-LSTM deep-learning architecture. The paper reports **99.84% accuracy for binary classification** and **99.76% accuracy for the 12-class multiclass task**. This project targets the multiclass CNN-LSTM result because it is the more demanding experiment and corresponds to the `PAPER_TARGET_ACCURACY = 0.9976` configured in the implementation.

The goal is **reproduction, not retrospective redesign**. Settings explicitly reported by the paper are retained wherever they are available. Missing details are implemented as explicit, inspectable reconstruction choices rather than being hidden. The raw CICDDoS2019 corpus is treated as read-only, and all generated artifacts remain inside the project output directory.

## Original Paper

> D. S. Rajput and A. K. Upadhyay, “Enhanced Network Defense: Optimized Multi-Layer Ensemble for DDoS Attack Detection,” *International Journal of Experimental Research and Review*, vol. 46, pp. 253–272, 2024. DOI: [10.52756/ijerr.2024.v46.020](https://doi.org/10.52756/ijerr.2024.v46.020).

The paper uses CICDDoS2019 and proposes a CNN-LSTM model in which convolutional layers perform spatial/feature extraction and LSTM layers model sequential dependencies. The publication reports the CNN-LSTM as the strongest deep-learning approach, with the multiclass experiment reaching **99.76% accuracy**.

## Reproduction Objective

The repository reproduces the paper's best **12-class multiclass CNN-LSTM** path as closely as the publication permits. It is designed to answer four practical questions:

1. Can the reported preprocessing and CNN-LSTM methodology be implemented end-to-end on CICDDoS2019?
2. How close can an independent execution get to the paper's 99.76% multiclass accuracy?
3. Which experiment details are directly supported by the publication, and which have to be reconstructed?
4. Can the same code run in constrained Apple-Silicon memory and in a high-memory Linux GPU server without changing the experiment logic?

## Paper Methodology and Project Mapping

The paper describes a hybrid CNN-LSTM detector for CICDDoS2019. This implementation maps that methodology to an auditable pipeline while preserving training/test separation for preprocessing and SMOTE.

| Stage | Paper/reproduction intent | Implementation in this repository |
| --- | --- | --- |
| Dataset | CICDDoS2019 | Recursively discovers source CSV files under `--data-dir`; source files are read-only. |
| Multiclass task | 12-class DDoS classification | Uses the reconstructed 12-class mapping shown below. |
| Data ingestion | Large CICDDoS2019 flow corpus | Reads CSVs in configurable chunks instead of loading the raw corpus at once. |
| Sampling | Publication does not provide a complete reproducible sampling recipe | Supports bounded per-file/per-class sampling or full uncapped retention with `0`. |
| Split | Training, validation and test evaluation | Stratified `70% / 15% / 15%` split. |
| Missing values | Clean/preprocess traffic records | Median imputation is fit on the training split only, then applied to validation/test. |
| Standardization | Normalized model inputs | Z-score `StandardScaler` is fit on training only and reused for validation/test. |
| Class imbalance | SMOTE | Classic multiclass SMOTE is applied **only to the standardized training split**. |
| Deep feature extraction | CNN | One or two `Conv1D` stages followed by max pooling. |
| Temporal/sequence modeling | LSTM | LSTM receives the CNN feature maps. |
| Parallel representation | Hybrid architecture | CNN output also feeds a parallel flattened Dense branch. |
| Fusion | Combined learned features | Dense and LSTM branches are concatenated. |
| Classification | Multiclass Softmax | Final Dense Softmax over 12 reconstructed classes. |
| Loss | Multiclass optimization | Categorical cross-entropy. |
| Evaluation | Accuracy and class-level performance | Held-out test metrics, confusion matrix, classification report, and predictions. |

A crucial implementation rule is that **validation and test data never participate in SMOTE**, and both the imputer and scaler are fit using the training split only.

## Pipeline

```mermaid
flowchart TD
    A[RAW CICDDoS2019<br/>read-only] --> B[Discover all CSV files]
    B --> C[Stream CSVs in chunks]
    C --> D{Sampling mode}
    D -->|Mac / bounded| E[Per-file and global class caps]
    D -->|Linux / full| F[Retain all target rows<br/>caps = 0]
    E --> G[Reconstructed 12-class dataset]
    F --> G
    G --> H[Stratified split<br/>70% train / 15% validation / 15% test]
    H --> I[Median imputer<br/>fit TRAIN only]
    I --> J[Z-score StandardScaler<br/>fit TRAIN only]
    J --> K[SMOTE<br/>TRAIN only]
    K --> L[One-hot 12-class labels]
    L --> M[Conv1D + MaxPooling]
    M --> N1[Flatten + Dense branch]
    M --> N2[LSTM branch]
    N1 --> O[Concatenate]
    N2 --> O
    O --> P[Post-merge Dense]
    P --> Q[12-class Softmax]
    Q --> R[Categorical cross-entropy training]
    R --> S[Validation during training]
    S --> T[Held-out TEST evaluation]
```

The same sequence in compact form is:

```text
RAW CICDDoS2019
→ chunked ingestion
→ bounded or full target-row retention
→ 70/15/15 stratified split
→ TRAIN-fitted median imputation
→ TRAIN-fitted z-score standardization
→ TRAIN-only SMOTE
→ one-hot labels
→ CNN
→ parallel Dense + LSTM
→ concatenation
→ Dense
→ 12-class Softmax
→ categorical cross-entropy
→ validation
→ held-out test evaluation
```

## Target Classes

The publication states a 12-class task but does not provide every implementation detail needed to reconstruct the exact class mapping. The project therefore uses the mapping already documented in the reproduction code:

1. `BENIGN`
2. `DrDoS_DNS`
3. `DrDoS_LDAP`
4. `DrDoS_MSSQL`
5. `DrDoS_NetBIOS`
6. `DrDoS_NTP`
7. `DrDoS_SNMP`
8. `DrDoS_SSDP`
9. `DrDoS_UDP`
10. `Syn`
11. `TFTP`
12. `UDP-lag`

`WebDDoS` and `Portmap`/`PortScan` are not part of this reconstructed default 12-class target.

## CNN-LSTM Architecture

The publication does not disclose every width, kernel, sequence-construction, optimizer, epoch, and SMOTE parameter needed for an exact independent rerun. The project therefore exposes these values through the CLI and records them in `config.json`.

The current reconstruction defaults are:

| Component | Default |
| --- | ---: |
| First `Conv1D` filters | 64 |
| Second `Conv1D` filters | 128 |
| Kernel size | 3 |
| Pool size | 2 |
| Dense branch units | 128 |
| LSTM units | 64 |
| Post-merge Dense units | 64 |
| Dropout | 0.0 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical cross-entropy |
| Batch size | 256 |
| Epochs | 30 |
| SMOTE neighbors | 5 |
| Early stopping | Disabled by default (`patience=0`) |

The input feature axis is treated as the CNN/LSTM sequence axis because the paper does not publish an exact temporal window length, stride, or flow grouping procedure.

## Dataset Handling and Sampling

The project expects a local copy of [CICDDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html).

### Read-only source protection

The raw dataset directory is never intentionally modified. The program records recursive CSV `size` and `mtime_ns` metadata before and after the experiment and raises an error if they differ.

### Bounded Mac mode

The `make run-mac` target is configured for a 16 GB-class Apple-Silicon machine:

```text
MAC_ROWS_PER_FILE_PER_CLASS=100000
MAC_GLOBAL_CLASS_CAP=100000
BATCH_SIZE=256
```

Positive sample caps use the existing seeded priority-reservoir logic. These values are Make variables and can be overridden without editing source code:

```bash
make run-mac MAC_ROWS_PER_FILE_PER_CLASS=50000 MAC_GLOBAL_CLASS_CAP=50000 BATCH_SIZE=128
```

### Full Linux mode

The `make run-linux` target is configured for a high-memory Linux GPU server:

```text
LINUX_ROWS_PER_FILE_PER_CLASS=0
LINUX_GLOBAL_CLASS_CAP=0
BATCH_SIZE=256
```

For this project, `0` has explicit retain-all/no-cap semantics:

```text
--rows-per-file-per-class 0  = retain every target row from every discovered CSV
--global-class-cap 0         = do not globally cap any target class
```

`--batch-size` affects TensorFlow training batches only; it does not limit dataset sampling. The Linux data and output defaults remain inside the expected server layout and repository directory, respectively.

## Project Structure

```text
CNN-LSTM-DDoS-Detection-CICDDoS2019/
├── .assets/
│   └── Sounds/
│       └── NotificationSound.wav
├── logs/
│   └── .gitkeep
├── .gitignore
├── LICENSE
├── Logger.py
├── Makefile
├── README.md
├── main.bib
├── main.py
├── requirements.txt
└── cnn_lstm_ddos_detection/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── constants.py
    ├── evaluation.py
    ├── experiment.py
    ├── model.py
    ├── persistence.py
    ├── preprocessing.py
    ├── sampling.py
    ├── schema.py
    ├── system.py
    ├── timing.py
    └── workflow.py
```

`main.py` remains the orchestrator. It now configures the repository-root `Logger.py`, registers the bundled completion sound with `atexit`, starts timing, parses the CLI, validates paths/options, and delegates to the package workflow. The experiment stages remain separated so the reproduction can be inspected and audited independently.

## Requirements

- Python **3.11 or 3.12**.
- GNU Make or a compatible `make` implementation.
- CICDDoS2019 available locally.
- macOS Apple Silicon or Linux.
- For Linux GPU execution: a working NVIDIA driver visible to TensorFlow.
- Sufficient storage for generated sampled arrays, transformed datasets, models, predictions, reports, and logs.
- Full-source execution can require substantial RAM, storage, and compute time—particularly the exact SMOTE nearest-neighbor stage.
- macOS completion playback uses the built-in `afplay` command.
- Linux completion playback optionally uses `aplay`; missing utilities or headless audio devices do not fail the experiment.

The platform-aware `requirements.txt` installs TensorFlow 2.18.1, `tensorflow-metal` 1.2.0 on Apple Silicon, TensorFlow CUDA user-space dependencies on Linux x86_64, NumPy, pandas, scikit-learn, matplotlib, joblib, and psutil. `Logger.py` and sound playback use only the Python standard library and operating-system commands.

## Setup

### Clone the repository

```bash
git clone https://github.com/BrenoFariasdaSilva/CNN-LSTM-DDoS-Detection-CICDDoS2019.git
cd CNN-LSTM-DDoS-Detection-CICDDoS2019
```

No separate installation script is required. The Makefile creates `.venv`, upgrades `pip`, `setuptools`, and `wheel`, installs the platform-aware `requirements.txt`, verifies TensorFlow GPU availability, and executes the experiment.

A manual environment remains supported:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py --help
```
