""" 			  		 			     			  	   		   	  			  	
Optimizer base. 

-----do not edit anything above this line---
"""


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        """

        # J = LCE + (reg/2)(sum of W-square)
        # gradient of J, equal gradient of LCE (which already done in part 2.2) plus derivative of L2
        # d_L2 = reg * Weights

        # loop through the hyperparameters in the model
        for key in model.gradients:
            # only apply regularization to weights W, not bias b
            if 'W' in key:
                model.gradients[key] += self.reg * model.weights[key]


