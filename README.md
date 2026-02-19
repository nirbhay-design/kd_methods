# Knowledge Distillation Methods

This repository contains two knowledge distillation methods operating on the CIFAR-10 dataset:
1. [**Naive KD (`kd.py`)**](https://arxiv.org/pdf/1503.02531)
2. [**Similarity Preserving Distillation (`spd.py`)**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf)

## Architecture Details
- **Teacher**: ResNet-18/50
- **Student**: A lightweight 3-layer Convolutional Neural Network.

## Usage
Simply run the shell script to execute both training pipelines sequentially:

```bash
bash run.sh
```