"""
ReLU Module.  

-----do not edit anything above this line---
"""

import numpy as np


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from relu.py!")

class ReLU:
    """
    An implementation of rectified linear units(ReLU)
    """

    def __init__(self):
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        The forward pass of ReLU. Save necessary variables for backward
        :param x: input data
        :return: output of the ReLU function
        '''
        out = None

        # The formula is y = x if x>0 and 0 if x<0.
        # Regardless of dimensionality out will have same shape as x, and np.maximum take care of ReLU function
        out = np.maximum(0,x)

        # store input x in cache for back prop
        self.cache = x
        return out

    def backward(self, dout):
        """
        :param dout: the upstream gradients
        :return:
        """
        dx, x = None, self.cache
        
        # dL/dx = dL/dy * dy/dx
        # dy/dx = 1 if x >0 and 0 if x < 0
        # dL/dy = dout. So gradient self.dx = dout if x >0 and 0 if x < 0
        dx = dout * (x > 0)

        self.dx = dx
