import numpy as np
from activation import activation_functions
from feedforward import forward

# calculate gradient of weights and biases (backpropagation)
def compute_grads(x, y, num_layers, weights, biases, activation):
  z, a = forward(x, num_layers, weights, biases, activation) # z---preactivation, a----activation
  y_cap = a[-1]
  delta = y_cap - y        # del(L)/del(z)
  grads_W, grads_b = [], []

  activation_derivative = activation_functions[activation][1]

  for k in reversed(range(len(weights))):
      grads_W.insert(0, np.outer(a[k], delta))
      grads_b.insert(0, delta)

      error = np.dot(delta, weights[k].T) # del(L)/del(a)
      if k > 0:
        delta = activation_derivative(z[k-1]) * error   # del(L)/del(z)

  return grads_W, grads_b

def compute_accuracy(Y_true, Y_pred):
  true_values = np.argmax(Y_true, axis=1)
  pred_values = np.argmax(Y_pred, axis=1)  
  return np.mean(true_values == pred_values)