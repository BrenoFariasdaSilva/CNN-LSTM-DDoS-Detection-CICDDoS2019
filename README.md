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
