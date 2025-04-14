"""Layer of neurons that contain a set of tensors. 
Keep track of the ability to run and train."""

import numpy as np

from duq_ds3_2025.cuppajoe import tensor

class Layer():
    def __init__(self):
        self.w = tensor.Tensor
        self.b = tensor.Tensor
        self.x = None # inputs to the layer
        self.grad_w = 0 # gradient weights
        self.grad_b = 0 # gradient bias vector
    
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """A forward pass through the layer"""
        raise NotImplementedError
    
    def backward(self, x:tensor.Tensor) -> tensor.Tensor:
        """A training/back-propagation pass through the layer"""
        raise NotImplementedError
    
class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        """Create a new linear layer"""
        super().__init__() # run init from parent class Layer
        self.w = np.random.randn(input_size, output_size) #matrix
        self.b = np.random.randn(output_size)        #vector 
    
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """The forward computation: y = w*x + b"""
        self.x = x
        return self.x @ self.w + self.b # @ is array multiplication

    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Computing derivative on data passing backwards through the network to figure out 
        the step necessary to take to train the network.
        
        X = wx + b
        y = f(x)
        dy/dw = f'(X)*x
        dy/dx = f'(X)*w
        dy/db = f'(x)
        
        The new component being added to variables in tensor form:
        if y = f(x) and X = x @ w + b and f'(x) is the gradient then
        dy/dx = f'(X) @ x.T (Transpose)
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """

        self.grad_b = np.sum(grad, axis = 0)
        self.grad_w = self.x.T @ grad

        return grad @ self.w.T
    
class Tanh(Linear):
    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x
        return np.tanh(super().forward(x)) # using Linear forward
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        grad = super().backward(grad)
        y = np.tanh(grad)

        return 1-y**2 # first derivative of tanh



