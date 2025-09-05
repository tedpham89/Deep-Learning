"""
CovNet Module.  

-----do not edit anything above this line---
"""

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from conv_classifier.py!")

class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None

        #Set initial input to x
        out = x
        # Loop through all model/layer in order, output of current model is input of the next
        for m in self.modules:
            out = m.forward(out)
        
        # collect probablity in loss using criterion softmax cross entropy
        probs, loss = self.criterion.forward(out,y)
        
        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        # init back pass on softmax cross entropy
        self.criterion.backward()
        # collect first upstream gradient
        dout = self.criterion.dx

        # Loop through each model/layer in reverse order to calculate input gradients, using it as upstream gradient of the next
        # Update dx,dw,db if available along the way
        for i in reversed(range(len(self.modules))):
            self.modules[i].backward(dout)
            dout = self.modules[i].dx
