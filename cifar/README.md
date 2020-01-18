# CIFAR-10 experiments

This folder contains the code for compressing CIFAR-10 models with structural pruning. The code support multiple common model architectures and layer configurations used on CIFAR-10. The pruning code is tested with ResNet models of various number of layers, and should also work with other architectures like VGG or DenseNet

## Usage

As described in Section 4.3 of the paper, the full training pipeline consist of training with sparsity-inducing regularizer, pruning and finetuning. Although the training pipeline can be started from scratch, based on common practice we suggest starting the training from a pretrained dense model to get better pruning performance.

### Acquiring pretrained dense model (optional)

```
python element.py --reg 0 --decay 0.0
```

By default the model will be trained for 250 epochs with learning rate 0.001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively.

The pretrained MLP model should reach around 98.4% testing accuracy, and CNN model should reach around 99.2% testing accuracy. The pretrained model will be saved to `saves/elt_0.0_0.pth`, which will be loaded later on in the sparse training step. 

### Training with sparsity-inducing regularizer

#### Element-wise pruning

```
python element.py --reg [regularization type] --decay [regularization strength] --pretrained
```

By default the model will be trained for 250 epochs with learning rate 0.001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively. The valid input to the `--reg` flag are integers from 0 to 4, which stands for 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1 respectively. Use the `--pretrained` flag if pretrained dense model is available.

The trained model will be saved to path `'saves/elt_(args.decay)_(args.reg_type).pth'`. Note that at this stage the model haven't been pruned yet, so it should still be a dense model but with a lot of close-to-zero elements.

#### Structural pruning

```
python structure.py --reg [regularization type] --decay [regularization strength] --pretrained
```

By default the model will be trained for 250 epochs with learning rate 0.001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively. The valid input to the `--reg` flag are integers from 0 to 2, which stands for 0:None 1:Group Lasso (L1) 2:Group-HS respectively. Use the `--pretrained` flag if pretrained dense model is available. The grouping is based on channels and filters.

The trained model will be saved to path `'saves/str_(args.decay)_(args.reg_type).pth'`. Note that at this stage the model haven't been pruned yet, so it should still be a dense model but with a lot of filters/channels close to zero.


### Pruning, finetuning and evaluating pruned model

Pruning can be done based on a fixed threshold, or based on a fixed ratio to the std of each layer. In parctice using std based threshold is prefered for MNIST experiments. The pruned model is then finetuned with zero elements fixed.

To set a fixed threshold for pruning, use 
```
python prun_tune_T.py --model [path to trained model] --sensitivity [pruning threshold]
```

To set a fixed ratio of std for pruning, use 
```
python prun_tune_V.py --model [path to trained model] --sensitivity [ratio of threshold/std]
```

By default the model will then be finetuned for 100 epochs with learning rate 0.0001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively. Detailed sparsity information (for both element-wise and structural sparsity) of each layer and the whold model will be printed, as well as the best testing accuracy achieved during finetuning. The pruned and finetuned model will also be stored.

## Acknowledgement
This code is adapted from [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).


# pytorch-classification
Classification on CIFAR-10/100 and ImageNet with PyTorch.

## Features
* Unified interface for different network architectures
* Multi-GPU support
* Training progress bar with rich info
* Training log and training curve visualization code (see `./utils/logger.py`)

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/bearpaw/pytorch-classification.git
  ```

## Training
Please see the [Training recipes](TRAINING.md) for how to train the models.

## Results

### CIFAR
Top1 error rate on the CIFAR-10/100 benchmarks are reported. You may get different results when training your models with different random seed.
Note that the number of parameters are computed on the CIFAR-10 dataset.

| Model                     | Params (M)         |  CIFAR-10 (%)      | CIFAR-100 (%)      |
| -------------------       | ------------------ | ------------------ | ------------------ |
| alexnet                   | 2.47               | 22.78              | 56.13              |
| vgg19_bn                  | 20.04              | 6.66               | 28.05              |
| ResNet-110                | 1.70               | 6.11               | 28.86              |
| PreResNet-110             | 1.70               | 4.94               | 23.65              |
| WRN-28-10 (drop 0.3)      | 36.48              | 3.79               | 18.14              |
| ResNeXt-29, 8x64          | 34.43              | 3.69               | 17.38              |
| ResNeXt-29, 16x64         | 68.16              | 3.53               | 17.30              |
| DenseNet-BC (L=100, k=12) | 0.77               | 4.54               | 22.88              |
| DenseNet-BC (L=190, k=40) | 25.62              | 3.32               | 17.17              |


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).
