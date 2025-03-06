import numpy as np
from activation import activation_functions
from feedforward import forward

# calculate gradient of weights and biases (backpropagation)
def compute_grads(x, y, num_layers, weights, biases, activation):
  z, a = forward(x, num_layers, weights, biases, activation)
  y_cap = a[-1]
  error = y_cap - y        # del(L)/del(y_cap) ------ w.r.to cross entropy loss function
  grads_W, grads_b = [], []

  activation_derivative = activation_functions[activation][1]

  for k in reversed(range(len(weights))):
      delta = error * activation_derivative(z[k])   # del(L)/del(z)
      grads_W.insert(0, np.outer(a[k], delta))
      grads_b.insert(0, delta)

      if k > 0:
          error = np.dot(delta, weights[k].T)      # del(L)/del(a)

  return grads_W, grads_b

def compute_accuracy(Y_true, Y_pred):
  true_values = np.argmax(Y_true, axis=1)
  pred_values = np.argmax(Y_pred, axis=1)  
  return np.mean(true_values == pred_values)