import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.flatten = nn.Flatten()
        self.Linear = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.Linear(x)
        return y