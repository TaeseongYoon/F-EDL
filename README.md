# Uncertainty Estimation by Flexible Evidential Deep Learning 
This repo contains an official PyTorch implementation for the paper [*Uncertainty Estimation by Flexible Evidential Deep Learning*](https://openreview.net/forum?id=N6ujq5Yfwa), accepted for publication in NeurIPS 2025

## Abstract 
Uncertainty quantification (UQ) is crucial for deploying machine learning models in high-stakes applications, where overconfident predictions can lead to serious consequences. An effective UQ method must balance computational efficiency with the ability to generalize across diverse scenarios. Evidential deep learning (EDL) achieves efficiency by modeling uncertainty through the prediction of a Dirichlet distribution over class probabilities. However, the restrictive assumption of Dirichlet-distributed class probabilities limits EDL's robustness, particularly in complex or unforeseen situations. To address this, we propose flexible evidential deep learning (
-EDL), which extends EDL by predicting a flexible Dirichlet distribution—a generalization of the Dirichlet distribution—over class probabilities. This approach provides a more expressive and adaptive representation of uncertainty, significantly enhancing UQ generalization and reliability under challenging scenarios. We theoretically establish several advantages of 
-EDL and empirically demonstrate its state-of-the-art UQ performance across diverse evaluation settings, including classical, long-tailed, and noisy in-distribution scenarios.

## Description
The main code for running F-EDL experiments is located in main.py. The script sequentially performs the following steps:
* Training
* Testing
* Misclassification detection
* OOD detection
* Distribution shift detection (applicable for CIFAR-10)


## How to Use
To run F-EDL experiments, execute the main.py script with the desired arguments. Below are example commands for different experimental setups:

1. Classical setting (CIFAR-10/CIFAR-100): python main.py --spect_norm
2. Long-tailed setting (mild imbalance, CIFAR-10-LT (rho = 0.1)): python main.py --imbalance_factor 0.1 --spect_norm 
3. Long-tailed setting (heavy imbalance, CIFAR-10-LT (rho = 0.01): python main.py --imbalance_factor 0.01 --spect_norm
4. Noisy setting (DMNIST): python main.py --ID_dataset "MNIST" --noise --spect_norm

Before running the code, please create the required directories: "./data", "./saved_results_fedl", and "./saved_models_fedl"
These directories are used to store datasets, experimental results, and trained model checkpoints, respectively. In addition, if a dataset is not downloaded automatically, please download it manually and place it in the appropriate subdirectory under "./data". 

You can customize hyperparameters and options, such as dataset, batch size, learning rate, dropout rate, and weight decay, based on your experimental preferences.


## Citation
If the code or the paper has been useful in your research, please consider citing our paper :
```latex
@article{yoon2025uncertainty,
  title={Uncertainty Estimation by Flexible Evidential Deep Learning},
  author={Yoon, Taeseong and Kim, Heeyoung},
  journal={arXiv preprint arXiv:2510.18322},
  year={2025}
}
```
