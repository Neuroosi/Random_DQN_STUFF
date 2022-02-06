import torch
from torch import nn
from torch._C import device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        super(NeuralNetwork, self).__init__()

        self.Q = nn.Sequential(nn.Linear(self.obsSpaceSize, 64),
        nn.ReLU(),
        nn.Linear(64, self.actionSpaceSize))

    def forward(self, x):
        x = x.to(device)
        Q_values = self.Q(x)
        return Q_values 