"""
MyModel model.  

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 2) #32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) #32x32x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)   # 32x16x16

        # Second convolution
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64x16x16
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) #64x16x16
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size from 64x8x8
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Adjusted for new spatial size
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 10)



    def forward(self, x):
        outs = None

        # Conv 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)  # (batch, 32, 8, 8)
        
        # Conv 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)  # (batch, 64, 4, 4)
        
        # Flatten for FC layers
        x = torch.flatten(x, start_dim=1)  # (batch, 64*4*4)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        outs = self.fc2(x)  #  output before CrossEntropyLoss 




        return outs
