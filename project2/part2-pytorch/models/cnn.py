"""
Vanilla CNN model.  

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from cnn.py!")


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()


        # Init layers
        # Convolution layer, input image of size 3x32x32, output 32x26x26   (32-7+1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        # Activation function relu does not change dimension
        self.relu = nn.ReLU()

        # maxpooling 2x2 reduce output to 32x13x13
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer will have input 32x13x13 from max pooling
        self.fc = nn.Linear(32 *  13* 13, 10)  

    def forward(self, x):
        outs = None
        # print("x shape", x.shape)

        # Let x went through convolution layer
        x  = self.conv1(x)
        # print("output shape after convo", x.shape)
        x = self.relu(x)
        # print("output shape after relu", x.shape)
        x = self.pool(x)
        # print("output shape after pool", x.shape)

        # X of size 32x13x13 need to be flatten before passing to fully connected layer
        x_flatten = torch.flatten(x, start_dim=1)

        # Fully connected layer
        outs = self.fc(x_flatten)


        return outs
