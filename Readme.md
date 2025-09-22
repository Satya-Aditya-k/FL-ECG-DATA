Federated Learning for Heterogeneous ECG Data

Privacy-preserving federated learning for 12-lead ECG classification across heterogeneous devices and sites.
This repo implements preprocessing, simulation of non-IID ECG clients, and federated training (FedAvg + heterogeneity-aware variants) to evaluate trade-offs between privacy and clinical performance. 

Key features

Simulation of realistic, heterogeneous ECG nodes from public datasets.

Modular preprocessing: resampling, denoising, normalization, feature extraction.

Federated training strategies (FedAvg, FedProx/FedDyn variants) and centralized baseline.

Evaluation with clinical metrics (Accuracy, F1, Sensitivity, Specificity) and convergence analysis.
