"""
Base  Optimizer.  

-----do not edit anything above this line---
"""

def hello_do_you_copy():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print("Roger that from _base_optimizer.py!")

class _BaseOptimizer:
  def __init__(self, model, learning_rate=1e-4, reg=1e-3):
      self.learning_rate = learning_rate
      self.reg = reg


  def update(self, model):
      pass

  def apply_regularization(self, model):
      '''
      Apply L2 penalty to the model. Update the gradient dictionary in the model
      :param model: The model with gradients
      :return: None, but the gradient dictionary of the model should be updated
      '''

      for m in model.modules:
          if hasattr(m, 'weight'):
              m.dw += self.reg * m.weight

