import numpy as np
from activation import activation_functions
from feedforward import forward

# calculate gradient of weights and biases (backpropagation)
def compute_grads(x, y, num_layers, weights, biases, activation, weight_decay):
  z, a = forward(x, num_layers, weights, biases, activation) # z---preactivation, a----activation
  y_cap = a[-1]
  delta = y_cap - y   #for cross entropy  # del(L)/del(z) 

  grads_W, grads_b = [], []

  activation_derivative = activation_functions[activation][1]

  for k in reversed(range(len(weights))):  
    # compute gradients with L2 regularization
    grad_W = np.outer(a[k], delta) + weight_decay * weights[k]
    grad_b = delta + weight_decay * biases[k]
    grads_W.insert(0, grad_W)
    grads_b.insert(0, grad_b)

    error = np.dot(delta, weights[k].T) # del(L)/del(a)
    if k > 0:
      delta = activation_derivative(z[k-1]) * error   # del(L)/del(z)

  return grads_W, grads_b