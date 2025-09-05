"""
SGD Optimizer.  

-----do not edit anything above this line---
"""

from ._base_optimizer import _BaseOptimizer
import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from sgd.py!")

class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # self.v = []
        # Init a velocity dictionary to store previous v
        self.v_dict = {}
        for idx, m in enumerate(model.modules):
            if hasattr(m,'weight'):
                self.v_dict[f'weight{idx}'] = np.zeros_like(m.weight)
            if hasattr(m,'bias'):
                self.v_dict[f'bias{idx}'] = np.zeros_like(m.bias)

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        # first step apply regularization reg
        self.apply_regularization(model)

        # loop through each model in order, and do gradient descending
        # velocity v with momentum is update along the way
        for idx, m in enumerate(model.modules):
            # this if filter out only model have object weight because ReLU and max_pool 
            # do not have parameter weight and bias and do not require gradient update
            if hasattr(m, 'weight'):
                v = self.v_dict[f'weight{idx}']
                v = v* self.momentum - self.learning_rate * m.dw
                m.weight += v
                self.v_dict[f'weight{idx}'] = v
            # same reason as above, update bias
            if hasattr(m,'bias'):
                v = self.v_dict[f'bias{idx}']
                v = v*self.momentum - self.learning_rate * m.db
                m.bias += v
                self.v_dict[f'bias{idx}'] = v

