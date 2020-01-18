import torch.nn as nn
import torch.nn.functional as F
import torch

from .prune import PruningModule, MaskedLinear


class EltwiseLayer(nn.Module):
  def __init__(self, n, train):
    super(EltwiseLayer, self).__init__()
    self.weights = nn.Parameter(torch.ones([1, n]), requires_grad=train)  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    return x * self.weights  # element-wise multiplication

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_act(PruningModule):
    def __init__(self, train=True):
        super(LeNet_act, self).__init__()
        linear = nn.Linear
        self.act1 = EltwiseLayer(784,train)
        self.fc1 = linear(784, 300)
        self.act2 = EltwiseLayer(300,train)
        self.fc2 = linear(300, 100)
        self.act3 = EltwiseLayer(100,train)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(self.act1(x)))
        x = F.relu(self.fc2(self.act2(x)))
        x = F.log_softmax(self.fc3(self.act3(x)), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else Linear
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        #self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(800, 500)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        #x = self.conv3(x)
        #x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
