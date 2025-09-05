"""
Focal Loss Wrapper.  

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """


    per_cls_weights = None

    # print("CLS_NUM_LIST",cls_num_list)
    # First calculate the effective number
    E_n = np.array([(1 - beta**n)/ (1-beta) if n > 0 else 0 for n in cls_num_list])
    # print("En", E_n)
    
    # Invers the effective number
    E_n = 1.0 / E_n

    # Calculate per class weight and normalize
    per_cls_weights = E_n/np.sum(E_n) * len(cls_num_list)
    per_cls_weights = torch.tensor(per_cls_weights,dtype = torch.float32)

    # Original set up e_n as np array but it seem local test require tensor
    # print(per_cls_weights)
    # print("CHECK",torch.is_tensor(per_cls_weights))

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        # print("TARGET", target)
        # Calculate log prob and probability from input
        log_probabilities = F.log_softmax(input, dim = 1)
        probabilities = torch.exp(log_probabilities)
        # print("PROB", probabilities)

        # Using the formula from 4.3 CLass-Balanced Focal Loss
        # Source: Class-Balanced Loss Based on Effective number of samples
        # First collecting true label probability
        target_prob = probabilities[torch.arange(len(target)), target]
        target_log_prob = log_probabilities[torch.arange(len(target)), target]
        # Appy the formula
        focal_loss = -(1 - target_prob)**self.gamma * target_log_prob
        
        # Apply weight
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[target]
        
        # Convert to scalar
        focal_loss = focal_loss.mean()
        # focal_loss = focal_loss.sum()

        loss = focal_loss

        return loss
