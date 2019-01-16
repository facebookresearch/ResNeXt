# ResNeXt: Aggregated Residual Transformations for Deep Neural Networks

By [Saining Xie](http://vcl.ucsd.edu/~sxie), [Ross Girshick](http://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Zhuowen Tu](http://pages.ucsd.edu/~ztu/), [Kaiming He](http://kaiminghe.com)

UC San Diego, Facebook AI Research

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements and Dependencies](#requirements-and-dependencies)
0. [Training](#training)
0. [ImageNet Pretrained Models](#imagenet-pretrained-models)
0. [Third-party re-implementations](#third-party-re-implementations)

#### News
* Congrats to the ILSVRC 2017 classification challenge winner [WMW](http://image-net.org/challenges/LSVRC/2017/results).
ResNeXt is the foundation of their new SENet architecture (a **ResNeXt-152 (64 x 4d)** with the Squeeze-and-Excitation module)!
* Check out Figure 6 in the new [Memory-Efficient Implementation of DenseNets](https://arxiv.org/pdf/1707.06990.pdf) paper for a comparision between ResNeXts and DenseNets. <sub>（*DenseNet cosine is DenseNet trained with cosine learning rate schedule.*）</sub>
<p align="center">
<img src="http://vcl.ucsd.edu/resnext/resnextvsdensenet.png" width="480">
</p>


### Introduction

This repository contains a [Torch](http://torch.ch) implementation for the [ResNeXt](https://arxiv.org/abs/1611.05431) algorithm for image classification. The code is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

[ResNeXt](https://arxiv.org/abs/1611.05431) is a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width.


![teaser](http://vcl.ucsd.edu/resnext/teaser.png)
##### Figure: Training curves on ImageNet-1K. (Left): ResNet/ResNeXt-50 with the same complexity (~4.1 billion FLOPs, ~25 million parameters); (Right): ResNet/ResNeXt-101 with the same complexity (~7.8 billion FLOPs, ~44 million parameters).
-----

### Citation
If you use ResNeXt in your research, please cite the paper:
```
@article{Xie2016,
  title={Aggregated Residual Transformations for Deep Neural Networks},
  author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}
```

### Requirements and Dependencies
See the fb.resnet.torch [installation instructions](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Download the [ImageNet](http://image-net.org/download-images) dataset and [move validation images](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to labeled subfolders

### Training

Please follow [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for the general usage of the code, including [how](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) to use pretrained ResNeXt models for your own task.

There are two new hyperparameters need to be specified to determine the bottleneck template:

**-baseWidth** and **-cardinality**

### 1x Complexity Configurations Reference Table

| baseWidth | cardinality |
|---------- | ----------- |
| 64        | 1           |
| 40        | 2           |
| 24        | 4           |
| 14        | 8           |
| 4         | 32          |


To train ResNeXt-50 (32x4d) on 8 GPUs for ImageNet:
```bash
th main.lua -dataset imagenet -bottleneckType resnext_C -depth 50 -baseWidth 4 -cardinality 32 -batchSize 256 -nGPU 8 -nThreads 8 -shareGradInput true -data [imagenet-folder]
```

To reproduce CIFAR results (e.g. ResNeXt 16x64d for cifar10) on 8 GPUs:
```bash
th main.lua -dataset cifar10 -bottleneckType resnext_C -depth 29 -baseWidth 64 -cardinality 16 -weightDecay 5e-4 -batchSize 128 -nGPU 8 -nThreads 8 -shareGradInput true
```
To get comparable results using 2/4 GPUs, you should change the batch size and the corresponding learning rate:
```bash
th main.lua -dataset cifar10 -bottleneckType resnext_C -depth 29 -baseWidth 64 -cardinality 16 -weightDecay 5e-4 -batchSize 64 -nGPU 4 -LR 0.05 -nThreads 8 -shareGradInput true
th main.lua -dataset cifar10 -bottleneckType resnext_C -depth 29 -baseWidth 64 -cardinality 16 -weightDecay 5e-4 -batchSize 32 -nGPU 2 -LR 0.025 -nThreads 8 -shareGradInput true
```
Note: CIFAR datasets will be automatically downloaded and processed for the first time. Note that in the arXiv paper CIFAR results are based on pre-activated bottleneck blocks and a batch size of 256. We found that better CIFAR test acurracy can be achieved using original bottleneck blocks and a batch size of 128.

### ImageNet Pretrained Models
ImageNet pretrained models are licensed under CC BY-NC 4.0.

[![CC BY-NC 4.0](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

#### Single-crop (224x224) validation error rate
| Network             | GFLOPS | Top-1 Error |  Download   |
| ------------------- | ------ | ----------- | ------------|
| ResNet-50 (1x64d)   |  ~4.1  |  23.9        | [Original ResNet-50](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)       |
| ResNeXt-50 (32x4d)  |  ~4.1  |  22.2        | [Download (191MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_50_32x4d.t7)       |
| ResNet-101 (1x64d)  |  ~7.8  |  22.0        | [Original ResNet-101](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained)      |
| ResNeXt-101 (32x4d) |  ~7.8  |  21.2        | [Download (338MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_101_32x4d.t7)      |
| ResNeXt-101 (64x4d) |  ~15.6 |  20.4        | [Download (638MB)](https://dl.fbaipublicfiles.com/resnext/imagenet_models/resnext_101_64x4d.t7)       |

### Third-party re-implementations

Besides our torch implementation, we recommend to see also the following third-party re-implementations and extensions:

1. Training code in PyTorch [code](https://github.com/prlz77/ResNeXt.pytorch)
1. Converting ImageNet pretrained model to PyTorch model and source. [code](https://github.com/clcarwin/convert_torch_to_pytorch)
1. Training code in MXNet and pretrained ImageNet models [code](https://github.com/dmlc/mxnet/tree/master/example/image-classification#imagenet-1k)
1. Caffe prototxt, pretrained ImageNet models (with ResNeXt-152), curves [code](https://github.com/cypw/ResNeXt-1)[code](https://github.com/terrychenism/ResNeXt)
