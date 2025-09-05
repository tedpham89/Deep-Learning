""" 			  		 			     			  	   		   	  			  	
MLP Model.  
-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None

        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError("check X shape and dimension, mismatch input size")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("check Y shape, mis match with X")

        # print("X",X.shape)
        #Forward Pass
        # Liner transformation X->(W1,b1)->c1
        c1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        # signmoid activation function
        a1 = self.sigmoid(c1)
        # Liner transformation a1 - >(W2,b2)->c2
        c2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        # print("c1",c1.shape, " a1 ",a1.shape, " c2 ", c2.shape)
        # a2 = self.sigmoid(c2)

        # softmax
        x_pred = self.softmax(c2)
        
        # Calculate cross entropy loss and accuracy
        loss = self.cross_entropy_loss(x_pred, y)
        accuracy = self.compute_accuracy(x_pred, y)

        if mode != 'train':
            return loss, accuracy


        # Backward calculation of gradient is the reverse of the foward process
        # logit -->ReLu-->softmax-->Cross Entropy loss.
        # THe gradient or derivative of softmax and cross entropyloss with respect to ReLU is just Pij - Yij for correct class.
        # Source: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

        # Computational graph
        # X-(w1,b1)-c1-(sigmoid)-a1-(w2,b2)-c2-(softmax)-x_pred-L(CrossEloss)
        # Gradient with respect to softmax input/reLU output

        batch_size = X.shape[0]
        one_hot_y = np.zeros_like(x_pred)
        one_hot_y[np.arange(batch_size),y] = 1
        g_c2 = x_pred - one_hot_y
        g_c2 = g_c2/batch_size
        # print("g_c2",g_c2.shape)

        # Second gradient with respect to W2 and b2
        # print("a1", a1.T.shape)
        self.gradients['W2'] = np.dot(a1.T, g_c2)
        self.gradients['b2'] = np.sum(g_c2, axis = 0)
        # print(self.gradients['W2'].shape)
        # print(self.gradients['b2'].shape)

        # Sigmoid gradient respect to a1
        g_a1 = np.dot(g_c2, self.weights['W2'].T)
        # d_c1 with respect to c1 (input of sigmoid)
        g_c1 = g_a1 *  self.sigmoid_dev(c1)
        # gradients with respect to W1 and b1
        self.gradients['W1'] = np.dot(X.T,g_c1)
        self.gradients['b1'] = np.sum(g_c1, axis = 0)
        # print("W1", self.gradients['W1'].shape)
        # print("b1",self.gradients['b1'].shape)


        return loss, accuracy
