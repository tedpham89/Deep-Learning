import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder as BasicGenerator

class BasicDiscriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, output_dim=1, leaky=False):
        super(BasicDiscriminator, self).__init__()

        # 1. Implement a basic discriminator with one hidden layer:                
        # 2. use RelU (leaky=False) or LeakyReLU (leaky=True) based on leaky param     

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if leaky:
            self.act1 = nn.LeakyReLU()
        else:
            self.act1 = nn.ReLU()
        # Linear + sigmoid
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, x):   

        # 1. implement forward pass given input x. note that x here is pre-flattened   #
        # print("GAN x", x.shape)
        # x = torch.flatten(x)
        hidden = self.act1(self.fc1(x))
        out = self.act2(self.fc2(hidden))
        return out

class BasicLeakyGenerator(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(BasicLeakyGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, z):
        h = self.act1(self.fc1(z))
        return self.act2(self.fc2(h))
