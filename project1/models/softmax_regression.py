""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None

        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError("check X shape and dimension, mismatch input size")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("check Y shape, mis match with X")

        # Output before ReLU, since there is no bias term
        store = np.dot(X, self.weights['W1'])

        logits = self.ReLU(store)
        # softmax
        x_pred = self.softmax(logits)

        # Calculate cross entropy loss and accuracy
        loss = self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)

        # print("loss", loss)
        # print("accuracy", accuracy)

        if mode != 'train':
            return loss, accuracy
        
        # Backward calculation of gradient is the reverse of the foward process
        # logit -->ReLu-->softmax-->Cross Entropy loss.
        # THe gradient or derivative of softmax and cross entropyloss with respect to ReLU is just Pij - Yij for correct class.
        # Source: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

        # Gradient with respect to softmax input/reLU output

        batch_size = X.shape[0]
        one_hot_y = np.zeros_like(x_pred)
        one_hot_y[np.arange(batch_size),y] = 1
        g_post_reLU = x_pred - one_hot_y
        g_post_reLU = g_post_reLU/batch_size
        # print(g_post_reLU.shape)

        # Use chain rule for gradient pre_reLU
        # print(store.shape)
        g_pre_reLU = self.ReLU_dev(store) * g_post_reLU
        # print(g_post_reLU.shape)
        # Lastly, since there is no bias
        self.gradients['W1'] = np.dot(X.T, g_pre_reLU)
        # print("gradient softmax", self.gradients['W1'])

        return loss, accuracy
