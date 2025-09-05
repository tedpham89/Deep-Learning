"""
Two Layer Network Model.  

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from twolayer.py!")


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super().__init__()
        
        # Init layers
        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_size)
        # Second layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # Activation function sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = None

        #Flatten x
        x_1d = torch.flatten(x, start_dim=1)
        # print("XSHAPE",x.shape)
        # Forward pass
        out = self.fc1(x_1d)
        # print("out1shape",out.shape)
        out = self.sigmoid(out)
        # print("out2shape",out.shape)
        out = self.fc2(out)
        # print("out3shape",out.shape)

        return out
