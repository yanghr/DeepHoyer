import torch.nn as nn
import torch.nn.functional as F
import torch

from .prune import PruningModule, MaskedLinear

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

class EltwiseLayer_FC(nn.Module):
  def __init__(self, n, train):
    super(EltwiseLayer_FC, self).__init__()
    self.weights = nn.Parameter(torch.ones([1, n]), requires_grad=train)  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    return x * self.weights  # element-wise multiplication

device = torch.device("cuda")
class EltwiseLayer(nn.Module):
  def __init__(self, n, train):
    super(EltwiseLayer, self).__init__()
    self.weights = nn.Parameter(torch.ones([1,n,1,1]), requires_grad=train)  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    return x * self.weights  # element-wise multiplication


class LeNet_5_act(PruningModule):
    def __init__(self, train=True):
        super(LeNet_5_act, self).__init__()
        linear = nn.Linear
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.act1 = EltwiseLayer(20,train)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.act2 = EltwiseLayer(50,train)
        #self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.act3 = EltwiseLayer_FC(800,train)
        self.fc1 = linear(800, 500)
        self.act4 = EltwiseLayer_FC(500,train)
        self.fc2 = linear(500, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(self.act1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(self.act2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        #x = self.conv3(x)
        #x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 800)
        x = self.fc1(self.act3(x))
        x = F.relu(x)
        x = self.fc2(self.act4(x))
        x = F.log_softmax(x, dim=1)

        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
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
        x = x.view(-1, 800)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
