import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(BasicEncoder, self).__init__()

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latent vectors mu and logvar
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #    1. linear -> relu to get hidden represntation
        # Reduce dimension from 784 input dim to 400 hidden dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        #    2. from hidden representation we have two heads.                       
        #     one to produce mu and another to produce logvar. 
        #     mu and logvar both use the same latent dim.    
                                    
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x):   
        """ forward pass for vae encoder.
            args: 
                x: [N, input_dim]

            outputs:
                mu: [N, latent_dim]
                logvar: [N, latent_dim]
        """
        
        #    1. implement forward pass for encoder.    
        # From x (N, input_dim)) through FC to (N,hidden_dim) then ReLU (N,hidden_dim)
        # then to mu and logvar (N, latent_dim)
        hidden_layer = self.act1(self.fc1(x))
        mu = self.fc_mu(hidden_layer)
        logvar = self.fc_logvar(hidden_layer)

        #    2. return mu and logvar                                    
        return mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=256):

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latents

            students implement Basic VAE using encoder and decoder. 
            1. instantiate encoder and pass in variables.
            2. instantiate decoder and pass in variables.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #    1. instantiate encoder (from above) and decoder(imported) for vae.
        #       Only need to instantiate it with appropriate args                  #
        #############################################################################
        self.encoder = BasicEncoder(input_dim, hidden_dim, latent_dim)

        #class BasicDecoder(nn.Module):
        #    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        self.decoder = BasicDecoder(latent_dim, hidden_dim, input_dim)


    def reparameterize(self, mu, logvar):
        """
            args:
                mu: [N,latent_dim]
                logvar: [N, latent_dim]
            outputs:
                z: reparameterized representation [N, latent_dim]

        """


        #    1. compute std from log-variance. 
        #    2. sample epsilon
        #    3. compute reparameterization.    
        #  Taken formula from page 11/23 "Tutorial on Variational Autoencoders"
        #   g z = mu(X) + sigma^1/2(X) ∗ e                                 

        # standard dev is sigma^1/2(X) or exp(logvar)^1/2
        std = (torch.exp(logvar) ** 0.5)
        # epsilon is sample from N(0,1)
        eps = torch.randn_like(std)
        # use formula above
        z = mu + std * eps
    
        return z
    
    def encode(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation [N, latent_dim]
                mu: mean latent [N, latent_dim]
                logvar: log variance latent [N, latent_dim]
        """

        #    1. encode imput image (NOTE recall the vae takes a flattened input)
        # Flatten x
        x_flat = torch.flatten(x, start_dim = 1)     # [N, C*H*W]
        mu, logvar = self.encoder(x_flat)
        #    2. compute reparameterization
        z = self.reparameterize(mu, logvar)
        
        return (z, mu, logvar)
    
    def forward(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation
                mu: mean latent
                logvar: log variance latent
        """
        # Encoder
        z, mu, logvar = self.encode(x)

        # Decoder step
        out = self.decoder(z)

        return (out, mu, logvar)



    @torch.no_grad()
    def generate(self, z):
        """
            args:
                x: input [N, latent_dim]
            outputs:
                out: (N, input_dim)
        """
        out = None
        self.eval()
        out = self.decoder(z)
        return out
