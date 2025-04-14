"""Create framework for a complete neural network"""

from duq_ds3_2025.cuppajoe import batch, loss, layer, mlp, tensor, optimizer

def train(nn: mlp.MLP,
          features: tensor.Tensor,
          labels: tensor.Tensor,
          epochs: int = 5000,
          iterator = batch.BatchIterator(),
          loss: loss.Loss = loss.MSE(),
          optimizer: optimizer.Optimizer = optimizer.SGD,
          learning_rate: float = 0.05):
    """Train a fully connected feedforward multilayer perceptron neural net"""
    optim = optimizer(nn, learning_rate)
    for e in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(features, labels):
            predictions = nn.forward(batch[0])
            epoch_loss += loss.loss(predictions, batch[1])
            grad = loss.grad(predictions, batch[1])

            nn.backward(grad)
            optim.step()
            nn.zero_parameters()
        print(f'Epoch {e} has loss {epoch_loss}')

if __name__ == '__main__':
    import numpy as np
    
    features = np.array([[0,0],[1,0],[0,1],[1,1]])
    labels = np.array([[1,0],[0,1],[0,1],[1,0]])

    network = mlp.MLP([layer.Tanh(2,2), layer.Tanh(2,2), layer.Tanh(2,2)])

    train(network, features, labels)

    print(network.forward(features))