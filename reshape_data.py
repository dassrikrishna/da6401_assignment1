import numpy as np

def transform(x):
    # normalize pixel values to [0,1] range
    x = x / 255.0
    # reshape the data
    x = x.reshape(-1, 784)
    return x

def one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

