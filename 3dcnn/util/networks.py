
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
            nn.ReLU(),
    
            nn.Conv3d(32, 32, kernel_size=3,padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)))
        self.group3 = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3,padding=1),
            nn.InstanceNorm3d(1))

        self.group4 = nn.Sequential(
            nn.Linear(22,20))
        
        self.group5 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(7,1,1),padding=0),
            nn.InstanceNorm3d(1),
            nn.Conv3d(1, 1, kernel_size=(7,1,1),padding=0),
            nn.InstanceNorm3d(1))
        
        self.group6 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3,padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)))
        
    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group2(out)
        out = self.group3(out)
        
        out = torch.permute(out, (0,1,3,4,2))
        out = self.group4(out)
        out = torch.permute(out, (0,1,4,2,3))
        
        #out = self.group5(out)



        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = CNNNet(**kwargs)
    return model
