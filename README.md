# DeepHoyer
This repo holds the codes for [DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures](https://openreview.net/pdf?id=rylBK34FDS). 

```
@inproceedings{
yang2020deephoyer,
title={DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant Sparsity Measures},
author={Huanrui Yang and Wei Wen and Hai Li},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rylBK34FDS}
}
```

The codes for MNIST, CIFAR-10 and ImageNet experiments are within `mnist/`, `cifar/` and `imagenet/` folder respectively. Please follow the README file in each folder to run the experiments. Codes are tested with Pytorch 1.2.0 and Python 3.6.8.


# Acknowledgement
The codes of the MNIST experiments are adapted from [Deep-Compression-PyTorch](https://github.com/mightydeveloper/Deep-Compression-PyTorch).
The codes of the CIFAR-10 experiments are adapted from [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification).
The codes of the MNIST experiments are adapted from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).
