# MNIST experiments

This folder contains the code for compressing MNIST models with element-wise or structural pruning. The code in `MLP/` and `CNN/` folders follows the same structure and usage. The code in `MLP/` aims to prune the LeNet-300-100 MLP model, and the code in `CNN/` aimes to prune the LeNet-5 CNN model.

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

By default the model will be trained for 250 epochs with learning rate 0.001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively. The valid input to the `--reg` flag are integers from 0 to 4, which stands for 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1 respectively.

The trained model will be saved to path `'saves/elt_(args.decay)_(args.reg_type).pth'`. Note that at this stage the model haven't been pruned yet, so it should still be a dense model but with a lot of close-to-zero elements.

#### Structural pruning

```
python structure.py --reg [regularization type] --decay [regularization strength] --pretrained
```

By default the model will be trained for 250 epochs with learning rate 0.001. These hyperparameters can be controlled with `--epochs` and `--lr` respectively. The valid input to the `--reg` flag are integers from 0 to 2, which stands for 0:None 1:Group Lasso (L1) 2:Group-HS respectively. The grouping is based on channels and filters.

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
This code is adapted from [Deep-Compression-PyTorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch).
