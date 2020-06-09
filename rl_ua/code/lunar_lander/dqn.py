import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
