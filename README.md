# Uncertainty Estimation by Flexible Evidential Deep Learning
This repo contains an official PyTorch implementation for the paper [*Uncertainty Estimation by Flexible Evidential Deep Learning*](https://openreview.net/forum?id=N6ujq5Yfwa), accepted for publication in NeurIPS 2025

## Abstract 
Uncertainty quantification (UQ) is crucial for deploying machine learning models in high-stakes applications, where overconfident predictions can lead to serious consequences. An effective UQ method must balance computational efficiency with the ability to generalize across diverse scenarios. Evidential deep learning (EDL) achieves efficiency by modeling uncertainty through the prediction of a Dirichlet distribution over class probabilities. However, the restrictive assumption of Dirichlet-distributed class probabilities limits EDL's robustness, particularly in complex or unforeseen situations. To address this, we propose flexible evidential deep learning (
-EDL), which extends EDL by predicting a flexible Dirichlet distribution—a generalization of the Dirichlet distribution—over class probabilities. This approach provides a more expressive and adaptive representation of uncertainty, significantly enhancing UQ generalization and reliability under challenging scenarios. We theoretically establish several advantages of 
-EDL and empirically demonstrate its state-of-the-art UQ performance across diverse evaluation settings, including classical, long-tailed, and noisy in-distribution scenarios.




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
