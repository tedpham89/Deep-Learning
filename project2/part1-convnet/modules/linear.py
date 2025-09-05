"""
Linear Module. 

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from linear.py!")

class Linear:
    """
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        """
        out = None

        # X (N,d1,d2...,dn). with in_dim = d1*d2*d3*...*dn
        # X flatten have shape (N, in_dim)
        batch_size = x.shape[0]
        # flatten X to (N, in_dim) shape
        x_flatten = x.reshape(batch_size,-1)
        # print("xshape",x_flatten.shape)

        # y = X.W + b
        # W is (in_dim,out_dim) shape
        # y is (N,out_dim) shape, and b is (out_dim,)
        
        out = np.dot(x_flatten, self.weight) + self.bias

        # Store inputs for backprop
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        """
        # Recalculate input dimension, batch size
        x = self.cache
        x_dim = x.shape
        batch_size = x.shape[0]
        # print("xdim", x_dim)
        x_flat = x.reshape(batch_size,-1)

        # The formula for gradient wrt w is dW = X.T * dY
        # Gradient wrt bias is dB = sum of dY , Y (N,out_dim), b is sum over row
        # gradient wrt X is dX = dY * W.T
        # x is flatten so need to reshape back to original dimension
        self.dw = np.dot(x_flat.T, dout)
        self.db = np.sum(dout, axis = 0)
        self.dx = np.dot(dout, self.weight.T).reshape(x_dim)

