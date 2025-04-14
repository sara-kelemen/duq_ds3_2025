"""A NN is a collection of layers; in this case, it is called a multilayer perceptron because
we are creating fully connected neural networks"""

from typing import Iterator
from duq_ds3_2025.cuppajoe import layer, tensor

class MLP():
    def __init__(self, layers: list[layer.Layer]):
        self.layers = layers

    def forward(self, x: tensor.Tensor ) -> tensor.Tensor:
        """Forward pass through the whole neural net"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Backward pass from a known gradient/error"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[tuple[tensor.Tensor, tensor.Tensor]]:
        """Return the weights and baises for each layer in turn, along with gradients"""
        for layer in self.layers:
            for pair in [(layer.w, layer.grad_w), (layer.b, layer.grad_b)]:
                yield pair

    def zero_parameters(self):
        """Set weights and biases to 0"""
        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0
