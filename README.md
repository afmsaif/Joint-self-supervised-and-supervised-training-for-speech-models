<p align="center">
  <img src="Screenshot 2023-09-07 at 21-23-28 AIRC_asr.png" width="500" title="BL-JUST Framework">
</p>

# Joint Unsupervised and Supervised Training for Automatic Speech Recognition via Bilevel Optimization

This repository contains the code for the paper **"Joint Unsupervised and Supervised Training for Automatic Speech Recognition via Bilevel Optimization"** by A F M Saif, Xiaodong Cui, Han Shen, Songtao Lu, Brian Kingsbury, and Tianyi Chen.

## Abstract

In this paper, we present a novel bilevel optimization-based training approach for acoustic models in automatic speech recognition (ASR) tasks, termed bi-level joint unsupervised and supervised training (BL-JUST). BL-JUST employs lower and upper level optimizations with unsupervised and supervised losses respectively, leveraging recent advances in penalty-based bilevel optimization to address this challenging ASR problem with manageable complexity and rigorous convergence guarantees. Extensive experiments on the LibriSpeech and TED-LIUM v2 datasets demonstrate that BL-JUST outperforms the commonly used pre-training followed by fine-tuning strategy.

## Key Contributions

1. **BL-JUST Framework**: Introduces a feedback loop between unsupervised and supervised training, unlike the conventional PT+FT strategy.
2. **Bilevel Optimization**: Utilizes penalty-based bilevel optimization for joint training with convergence guarantees.
3. **Empirical Results**: Demonstrates superior performance on LibriSpeech and TED-LIUM v2 datasets, reducing word error rates (WERs) and improving training efficiency.

## Contents

- **Code**: Implementation of the BL-JUST training framework.
- **Experiments**: Scripts and configurations for reproducing the experiments presented in the paper.
- **Datasets**: Instructions for downloading and preparing the LibriSpeech and TED-LIUM v2 datasets.

## Getting Started

1. **Dependencies**:
    - Python=3.9
    - Pytorch=2
2. **Installation**: Step-by-step guide to setting up the environment.
    ```bash
    git clone https://github.com/yourusername/bl-just-asr.git
    cd bl-just-asr
    pip install -r requirements.txt
    ```
3. **Running Experiments**: Detailed instructions to run the training and evaluation scripts.

## Usage

- **Training**: Example commands for training the ASR models using the BL-JUST framework.
- **Evaluation**: Commands to evaluate the trained models and reproduce the results from the paper.

## Results

- **Performance Metrics**: Summary of the ASR performance on different datasets.
- **Comparative Analysis**: Comparison with the traditional PT+FT approach.

## Citation

If you find this code useful in your research, please consider citing our paper:

@article{saif2024joint,
title={Joint Unsupervised and Supervised Training for Automatic Speech Recognition via Bilevel Optimization},
author={Saif, AFM and Cui, Xiaodong and Shen, Han and Lu, Songtao and Kingsbury, Brian and Chen, Tianyi},
journal={arXiv preprint arXiv:2401.06980},
year={2024}
}


## Acknowledgments

This work was supported by the Rensselaer-IBM AI Research Collaboration, part of the IBM AI Horizons Network and Cisco research grant.
