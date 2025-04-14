"""Loss functions are used to train a NN, measuring the difference between predictions and labels."""

import numpy as np

from duq_ds3_2025.cuppajoe import tensor

class Loss():
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float: 
        """From predictions and labels, figure out how wrong we are

        Returns:
            float: wrongness
        """
        raise NotImplementedError
    
    def grad(self, predictions:tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        """The gradient of loss function with respect to the predictions"""

        raise NotImplementedError
    
class MSE(Loss):
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float: 
        return np.mean((predictions-labels)**2) # single val
    
    def grad(self, predictions:tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        return 2*(predictions-labels) # vector