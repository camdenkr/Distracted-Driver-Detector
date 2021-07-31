import torch.nn as nn
import torch.nn.functional as F
import torch

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5)
        self.layer2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5)
        self.layer3 = torch.nn.Linear(3200, 16)
        self.layer4 = torch.nn.Linear(16, 10)

    def forward(self, x):
        # pass through conv layers
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)

        # pass through linear layers
        x = torch.flatten(x, start_dim=1)  # flatten the output of convolution
        x = self.layer3(x)
        x = torch.nn.functional.relu(x)
        x = self.layer4(x)
        return x
