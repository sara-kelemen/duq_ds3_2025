"""Need a mechanism for batching training data for the NN"""

import numpy as np

from duq_ds3_2025.cuppajoe import tensor

class DataIterator():
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor):
        """batch a set of features and labels"""
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        """Create a new iterator for batching data

        Args:
            batch_size (int, optional): number of training points to use together to calc grad. Defaults to 32.
            shuffle (bool, optional): shuffle training data into diff batches. Defaults to True.
        """ 
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor):
        starts = np.arange(0, len(features), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_features = features[start:end]
            batch_labels = labels[start:end]
            yield (batch_features, batch_labels) 

