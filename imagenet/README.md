# ImageNet experiments

This folder contains the code for compressing ImageNet models with structural pruning. The code is based on the official PyTorch tutorial for training ImageNet model ([pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet)). The pruning code is tested with ResNet models of various number of layers, and should also work with other architectures like VGG or DenseNet.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage

As described in Section 4.3 of the paper, the full training pipeline consist of training with sparsity-inducing regularizer, pruning and finetuning. Although the training pipeline can be started from scratch, based on common practice we suggest starting the training from a pretrained dense model to get better pruning performance.

### Acquiring pretrained dense model (optional)

PyTorch has officially provided the pretrained ImageNet models of various model structures in the [TorchVision model zoo](https://pytorch.org/docs/stable/torchvision/models.html). These pretrained models can be automatically downloaded and loaded by simplly set a `pretrained` parameter when declaring the model. So there's no need to do the training yourself. You may check [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet) if you want to train your own model from scratch.

### Training with sparsity-inducing regularizer

Here we apply group sparsity regularizations (group LASSO or group-HS) tot he training process of the ImageNet model

```
python main_group.py -a resnet50 --pretrained --lr 0.1 --epochs 90 --reg [regularization type] --decay [path to pretrained model] [imagenet-folder with train and val folders]
```

The pretrained model will be used if `--pretrained` flag is included.
The valid input to the `--reg` flag are integers from 0 to 2, which stands for 0:None 1:Group Lasso (L1) 2:Group-HS respectively. Additional to these hyperparameters, you may use `--lr_int` to set the learning rate decay interval (default as 30), and use `--lr_decay` to set the lr decay ratio (default as 0.1). Please see the full documentation of [pytorch/example](https://github.com/pytorch/examples/tree/master/imagenet#usage) if you are interested in tuning other hyperparameters not mentioned here.

The trained model will be saved to path `os.path.join('./results', str(args.arch)+'_'+str(args.decay)+'_'+str(args.reg), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))` with filename `model.pth`. Training logs and related learning curves are also stored in this path. Note that at this stage the model haven't been pruned yet, so it should still be a dense model but with a lot of filters/channels close to zero.


### Pruning, finetuning and evaluating pruned model

For ImageNet models we use a fixed threshold for pruning. The pruned model is then finetuned with all zero elements fixed.

To set a fixed threshold for pruning, use 

```
python prun_tune.py -a resnet50 --lr 0.01 --epochs 90 --resume [path to trained model] --sensitivity [pruning threshold] [imagenet-folder with train and val folders]
```

You may use `--lr_int` to set the learning rate decay interval (default as 30), and use `--lr_decay` to set the lr decay ratio (default as 0.1). If the sparsely trained model is not saved with default filename `model.pth`, you can use `--model` flag to indicate the correct filename.

The pruned and finetuned model will be saved to path `os.path.join(args.resume, 'finetune', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))` with filename `finetuned.pth`.
Detailed structural sparsity information of each layer will be printed, as well as the best testing accuracy achieved during finetuning. 
After the finetuning process is finished, you can evaluate the sparsity and performance of a finetuned model by using 

```
python prun_tune.py -a resnet50 --resume [path to finetuned model] --model 'finetuned' --sensitivity 0.0 --evaluate [imagenet-folder with train and val folders]
```
Note that since the finetuned model has already been pruned, the `--sensitivity` should be set to 0.0 during evaluation.

## Acknowledgement
This code is adapted from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).




