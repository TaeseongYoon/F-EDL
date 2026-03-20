# Uncertainty Estimation by Flexible Evidential Deep Learning

[![Paper](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=N6ujq5Yfwa)
[![Conference](https://img.shields.io/badge/NeurIPS-2025-success)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-informational)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-red)](#)

Official PyTorch implementation of the paper [*Uncertainty Estimation by Flexible Evidential Deep Learning*](https://openreview.net/forum?id=N6ujq5Yfwa), accepted for publication at **NeurIPS 2025**.

---

## Abstract

Uncertainty quantification (UQ) is crucial for deploying machine learning models in high-stakes applications, where overconfident predictions can lead to serious consequences. An effective UQ method should balance computational efficiency with the ability to generalize across diverse scenarios. Evidential deep learning (EDL) achieves computational efficiency by modeling uncertainty through a Dirichlet distribution over class probabilities. However, the restrictive assumption that class probabilities follow a Dirichlet distribution can limit the robustness of EDL, particularly in complex or unforeseen situations. To address this limitation, we propose **Flexible Evidential Deep Learning (F-EDL)**, which extends EDL by predicting a **flexible Dirichlet distribution**, a generalization of the Dirichlet distribution, over class probabilities. This approach enables a more expressive and adaptive representation of uncertainty, significantly improving UQ generalization and reliability in challenging scenarios. We theoretically establish several advantages of **F-EDL** and empirically demonstrate its state-of-the-art UQ performance across diverse evaluation settings, including classical, long-tailed, and noisy in-distribution scenarios.

---

## Overview

The main code for running F-EDL experiments is located in `main.py`. Running the script sequentially performs the following steps:

- Training
- Testing
- Misclassification detection
- OOD detection
- Distribution shift detection (applicable to CIFAR-10)

---

## Requirements

The codebase is implemented in **PyTorch**. Recommended environment:

- Python 3.9+
- PyTorch
- torchvision
- numpy
- scikit-learn
- scipy
- tqdm
- matplotlib
- pandas

Install the main dependencies with:

```bash
pip install torch torchvision numpy scipy scikit-learn tqdm matplotlib pandas
```

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Before running experiments, create the required directories:

```bash
mkdir -p ./data ./saved_results_fedl ./saved_models_fedl
```

These directories are used for:

- `./data`: datasets
- `./saved_results_fedl`: experimental results
- `./saved_models_fedl`: trained model checkpoints

If a dataset is not downloaded automatically, please download it manually and place it in the appropriate subdirectory under `./data`.

---

## How to Use

To run F-EDL experiments, execute `main.py` with the desired arguments.

### Example commands

- **(i) Classical setting (CIFAR-10 / CIFAR-100)**

```bash
python main.py --spect_norm
```

- **(ii) Long-tailed setting (mild imbalance, CIFAR-10-LT with rho = 0.1)**

```bash
python main.py --imbalance_factor 0.1 --spect_norm
```

- **(iii) Long-tailed setting (heavy imbalance, CIFAR-10-LT with rho = 0.01)**

```bash
python main.py --imbalance_factor 0.01 --spect_norm
```

- **(iv) Noisy setting (Dirty-MNIST / noisy MNIST setting)**

```bash
python main.py --ID_dataset MNIST --noise --spect_norm
```

---

## Configuration

You can customize various hyperparameters and options depending on your experimental setup, including:

- dataset
- batch size
- learning rate
- dropout rate
- weight decay
- spectral normalization
- imbalance factor
- noise setting

Please refer to `main.py` for the full list of supported arguments.

---

## Output

During execution, the code saves:

- trained model checkpoints to `./saved_models_fedl`
- experiment results to `./saved_results_fedl`

Depending on the experimental setting, the evaluation pipeline may include:

- classification accuracy
- misclassification detection
- OOD detection
- distribution shift detection

---

## Notes

- Distribution shift detection is currently applicable to **CIFAR-10**.
- For long-tailed experiments, use the `--imbalance_factor` argument.
- For noisy experiments, use the `--noise` flag.
- Spectral normalization can be enabled with `--spect_norm`.

---

## Citation

If this code or paper has been useful in your research, please consider citing our work:

```latex
@article{yoon2025uncertainty,
  title={Uncertainty Estimation by Flexible Evidential Deep Learning},
  author={Yoon, Taeseong and Kim, Heeyoung},
  journal={arXiv preprint arXiv:2510.18322},
  year={2025}
}
```

---

## Acknowledgement

Thank you for your interest in our work. We hope this repository is useful for research on uncertainty quantification and evidential deep learning.
