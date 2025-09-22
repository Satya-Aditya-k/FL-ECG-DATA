
**`PRNA/features/` — feature engineering (what to expect)**  
This folder contains the signal processing and feature-extraction pipeline used to transform raw ECG recordings into model-ready inputs. contents and responsibilities:
- resampling, filtering and denoising utilities  
- channel selection / normalization routines  
- time-domain & frequency-domain feature extraction scripts  
- scripts that read raw data and write preprocessed feature files (e.g., `.npy`, `.npz`, or CSV) consumed by training pipelines

Run the feature pipeline first to produce the dataset used by federated simulations.

**`PRNA/` — model building & training (what to expect)**  
This folder hosts the model definitions, training loops and evaluation utilities used for both federated and centralized experiments. Typical contents and responsibilities:
- model architectures and layer definitions (PyTorch modules)  
- configuration files or scripts to launch experiments (hyperparams, number of clients, rounds, etc.)

---
