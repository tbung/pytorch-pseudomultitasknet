import torch
import torch.nn as nn

import numpy as np

class NearestMean(nn.Module):
    def __init__(self, features, classes):
        super(self.__class__, self).__init__()

        self.features = features
        self.classes = classes

        # We need mean and variance per feature and class
        self.register_parameter(
            "means",
            nn.Parameter(torch.Tensor(self.classes, self.features))
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.means.data = torch.eye(self.classes, self.features)

    def forward(self, x):
        x = x[:,np.newaxis,:]
        dist = torch.sqrt(
            torch.sum(
                (x - self.means)**2, dim=-1
            )
        )
        return ((dist - torch.max(dist, -1, True)[0]) == 0).float()

