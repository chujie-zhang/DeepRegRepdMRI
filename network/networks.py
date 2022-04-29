
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np

class CNNNet(nn.Module):
    def __init__(self):

        super(CNNNet, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3,padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)))
        self.group2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3,padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3,padding=1),
            nn.InstanceNorm3d(1))
        self.group4 = nn.Sequential(
            nn.Linear(22,20))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group2(out)
        out = self.group3(out)

        out = torch.permute(out, (0,1,3,4,2))
        out = self.group4(out)
        out = torch.permute(out, (0,1,4,2,3))





        return out

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = CNNNet(**kwargs)
    return model
