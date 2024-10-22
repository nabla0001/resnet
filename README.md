# Deep Residual Networks

PyTorch implementation of the original CIFAR-10 Residual Net (ResNet) models published in
["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385) by He et al. (2015) [1].

## Overview

* [Introduction](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#introduction)
* [Motivation](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#motivation)
* [Results](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#results)
* [Hyperparameters](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#hyperparameters)
* [GPU training](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#gpu-training)
* [Usage](https://github.com/nabla0001/resnet/tree/main?tab=readme-ov-file#usage)

## Introduction

Published in 2015, the paper was a deep learning milestone 
showing how to train neural networks of unseen depth at the time. 
Residual Nets were able to stack 100 to 1000 layers while state-of-the-art models such as VGG-19 contained less than 20.

It achieved this by introducing "shortcut connections" between layers which forward
activations without transformations and improve optimisation. The resulting deep networks won the ImageNet Large Scale
Visual Recognition Challenge (ILSVRC) in 2015. Shortcut connections have become a foundational building block in deep learning
and are applied in many influential contemporary architectures such as
Transformers, UNets and EfficientNets.

## Motivation

The goals of this project were

1. understand the ResNet architecture in detail by building it
2. validate the implementation by re-producing the CIFAR-10 experiements from the paper (section 4.2.)


## Results

The implementation closely re-produces the results in [1].

### The degradation problem

The experiments re-produce Fig. 6 in the paper and show the "degradation problem" of making  networks deeper: 
curiously, as CNNs without residual connections (*Plain Nets*) are made deeper their *training error* (as well as their test error)
increases. So this effect is not the result of over-fitting. The deeper models clearly perform worse than their shallower 
counterparts. This issue also occurs for other datasets, e.g. ImageNet.

*Residual Nets*, on the other hand, address this problem and 
1. improve performance as network depth increases
2. show better final test performance than plain counterparts

![Fig. 6](/plots/training-curves.png)

### Test performance

| model      | test error (%) | test error (%) [*He et al.*] | # params | # layers |
|------------|----------------|------------------------------|----------|----------|
| ResNet-20  | 8.51  (±0.22)  | 8.75                         | 0.270M   | 20       |
| ResNet-56  | 7.40  (±0.27)  | 6.97                         | 0.853M   | 56       |
| ResNet-110 | 7.14  (±0.20)  | 6.61                         | 1.732M   | 110      | 
| Plain-20   | 9.78  (±0.13)  | ~9<sup>*</sup>               | 0.270M   | 20       | 
| Plain-56   | 12.95 (±0.22)  | ~13<sup>*</sup>              | 0.853M   | 56       | 

 <sup>**plain net errors are read from graphs because they are not reported*</sup>

For my experiments I report the mean (± std) across 5 runs.

## Hyperparameters

Training hyperparameters match [1].

| Parameter       | Value                                |
|-----------------|--------------------------------------|
| `optimiser`     | `SGD`                                | 
| `learning rate` | 0.1, divided by 10 @ batch 32k & 48k | 
| `batch_size`    | 128                                  | 
| `momentum`      | 0.9                                  | 
| `weight_decay`  | 1e-4                                 |

Following [1] for ResNet-110 the initial learning rate is 0.01 which is then increased
to 0.1 after 400 batches. All models are trained for 64k batches.

#### Weight initialisation

* Kaiming normal distribution for all `Conv2d` layers following [2]

#### Image pre-processing

* Pixels are normalised into [0, 1]
* Chanel-wise mean/std normalisation

#### Data augmentation

* Zero-padding: 4 pixels
* Random-crop: 32x32 pixels
* Random horizontal flip

## GPU training

The implementation supports GPU and CPU training and automatically
checks for available devices:

* `mps`
* `cuda`
* `cpu`

All models were trained on a M3 MacBook via _Metal Performance Shaders_ (MPS) backend.

Approximate training times for each model are:

| Model      | Training time |
|------------|---------------|
| ResNet-20  | 25 min        | 
| ResNet-56  | 65 min        | 
| ResNet-110 | 130 min       | 

## Usage

You can re-create my `conda` environment via

```shell
conda env create -f env.yml
```

To run experiments


```shell
conda activate pytorch

# ResNet20
python train.py --exp-name resnet20 --n 3 --skip-connection zeropad

# ResNet110
python train.py --exp-name resnet110 --n 18 --skip-connection zeropad

# PlainNet20
python train.py --exp-name resnet56 --n 3 --model-type plain
```
see `train.py` for all available command line options.

Each experiment produces two files: a results file (`.pkl`) and model checkpoint (`.ckpt`) which are written to
a subfolder `{exp_name}` in `experiments` (configurable via command line).

## Related work

* [Facebook AI Research (FAIR) resnet](https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/resnet.lua)
* [`torchvision` resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## References

[1] He, Zhang, Ren and Sun. 2015. ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385).

[2] He, Zhang, Ren, and Sun. 2015. ["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"](https://arxiv.org/pdf/1502.01852).