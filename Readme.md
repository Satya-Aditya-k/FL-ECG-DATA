# Federated Learning for Heterogeneous ECG Data

[![Repo](https://img.shields.io/badge/repo-FL--ECG--DATA-blue)](https://github.com/Satya-Aditya-k/FL-ECG-DATA/tree/main) ![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **Privacy-preserving federated learning for 12-lead ECG classification across heterogeneous devices and sites.**  
> Simulates different variants, IID ,non-IID ECG clients and also based on scaling(centralized/distributed scaling of data), harmonizes heterogeneous signals, and evaluates federated strategies vs. centralized baselines.

---

## 🔎 Short abstract
This repository provides code, notebooks and scripts to experiment with federated learning (FL) on multi-source 12-lead ECG datasets. It focuses on realistic heterogeneity (different sampling rates, noise, lengths and class distributions) and compares FL algorithms (FedAvg and heterogeneity-aware variants such as FedProx / FedDyn) against centralized baselines using clinically relevant metrics.

Repo: https://github.com/Satya-Aditya-k/FL-ECG-DATA/tree/main

---

## ✨ Key features
- **Heterogeneous client simulation** — build realistic non-IID nodes from public ECG collections.  
- **Modular preprocessing** — resampling, denoising, normalization, time/frequency features.  
- **Federated algorithms** — FedAvg, FedProx, FedDyn (and configurable variants).  
- **Evaluation suite** — Accuracy, F1, Sensitivity, Specificity and convergence analysis.  
- **Notebooks & scripts** — reproducible workflows for preprocessing → training → evaluation.

---

## 🚀 Quick start

```bash
# 1. clone
git clone https://github.com/Satya-Aditya-k/FL-ECG-DATA.git
cd FL-ECG-DATA

# 2. create env and install deps (example)
python -m venv .venv
# on Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. open main notebook for an end-to-end run
jupyter lab notebooks/
