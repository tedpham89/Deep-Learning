""" 			  		 			     			  	   		   	  			  	
Models Base. 

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np


class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        """
        prob = None

        # exp
        e = np.exp(scores)

        prob = e / np.sum(e, axis = 1, keepdims=True)

        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        """
        loss = None

        # Initiliza list to store loss of each sample
        batch_loss = []

        # Loop over each sample in batch
        for i in range(len(y)):
            correct_p = x_pred[i, y[i]]
            # cross entropy loss is -log(pi) where pi is prob of correct class
            cel = -np.log(correct_p)
            # print("crossEloss ",cel)
            batch_loss.append(cel)

        # calculate mean of all loss in the batch
        loss = np.mean(batch_loss)
        # print("loss ", loss)
        return loss

    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        acc = None

        # highest predicted probability is the predicted class, compare it with y, and calcualte accurary

        counter = 0

        # loop through each sample
        for i in range(len(y)):
            prediction = np.argmax(x_pred[i])

            if prediction == y[i]:
                counter += 1

        # Accuracy is probablity of correct prediction.
        acc = counter/len(y)
        # print("accuracy ", acc)


        return acc

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        """
        out = None

        # sigmoid function
        out = 1 / (1 + np.exp(-X))


        return out

    def sigmoid_dev(self, x):
        """
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        """
        ds = None

        # sigmoid function
        sig_func = 1 / (1 + np.exp(-x))
        # derivative
        ds = sig_func * ( 1 - sig_func)


        return ds

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        out = None

        # ReLU function
        out = np.maximum(0, X)
        # print(out.shape)

        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        out = None

        # derivative of relu is 0 if x smaller than 0, and 1 if x is higher than 0
        out = (X > 0) * 1.0
        # print(out.shape)

        return out
