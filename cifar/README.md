# CIFAR-10 experiments

This folder contains the code for compressing CIFAR-10 models with structural pruning. The code support multiple common model architectures and layer configurations used on CIFAR-10. The pruning code is tested with ResNet models of various number of layers, and should also work with other architectures like VGG or DenseNet

## Usage

As described in Section 4.3 of the paper, the full training pipeline consist of training with sparsity-inducing regularizer, pruning and finetuning. Although the training pipeline can be started from scratch, based on common practice we suggest starting the training from a pretrained dense model to get better pruning performance.

### Acquiring pretrained dense model (optional)

```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 
```

The initial learning rate is set by default to 0.1, which can be changed with `--lr` if needed.
The model will be saved in the indicated `--checkpoint` folder with filename `model_best.pth.tar`

For more details on training the dense model please see the training recipes provided by [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md).

### Training with sparsity-inducing regularizer

Here we apply group sparsity regularizations (group LASSO or group-HS) tot he training process of the CIFAR-10 model

```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --reg [regularization type] --decay [regularization strength] --model [path to pretrained model]
```

The valid input to the `--reg` flag are integers from 0 to 2, which stands for 0:None 1:Group Lasso (L1) 2:Group-HS respectively.

If you intend to load the pretrained model, please make sure the `-a` and `--depth` flag is set as the same as the architecture of the loaded model. You may first add `-e` to the command to evaluate the loaded model before training starts. 

The trained model will be saved to path `os.path.join('./results', str(args.arch)+'_'+str(args.decay)+'_'+str(args.reg), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))` with filename `model.pth`. Note that at this stage the model haven't been pruned yet, so it should still be a dense model but with a lot of filters/channels close to zero.


### Pruning, finetuning and evaluating pruned model

For CIFAR-10 models we use a fixed threshold for pruning. The pruned model is then finetuned with all zero elements fixed.

To set a fixed threshold for pruning, use 

```
python cifar_prun_tune.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --resume [path to trained model] --sensitivity [pruning threshold]
```

The initial learning rate is set by default to 0.1, which can be changed with `--lr` if needed.

The pruned and finetuned model will be saved to path `os.path.join(args.resume, 'finetune', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))` with filename `finetuned.pth`.
Detailed structural sparsity information of each layer will be printed, as well as the best testing accuracy achieved during finetuning. 
After the finetuning process is finished, you can evaluate the sparsity and performance of a finetuned model by using 

```
python cifar_prun_tune.py -a resnet --depth 110 --resume [path to finetuned model] --sensitivity 0.0 --evaluate
```
Note that since the finetuned model has already been pruned, the `--sensitivity` should be set to 0.0 during evaluation.

## Acknowledgement
This code is adapted from [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).



