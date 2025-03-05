import numpy as np
from activation import activation_functions
from feedforward import forward

# calculate gradient of weights and biases
def compute_grads(x, y, num_layers, weights, biases, activation):
  z, a = forward(x, num_layers, weights, biases, activation)
  y_cap = a[-1]
  error = y_cap - y        # del(L)/del(y_cap)
  grads_W, grads_b = [], []

  activation_derivative = activation_functions[activation][1]

  for k in reversed(range(len(weights))):
      delta = error * activation_derivative(z[k])   # del(L)/del(z)
      grads_W.insert(0, np.outer(a[k], delta))
      grads_b.insert(0, delta)

      if k > 0:
          error = np.dot(delta, weights[k].T)      # del(L)/del(a)

  return grads_W, grads_b

# define stochastic gradient descent
def sgd(X, Y, weights, biases, num_layers, learning_rate, epochs, activation):
  for _ in range(epochs):
    dw = [np.zeros_like(w) for w in weights]
    db = [np.zeros_like(b) for b in biases]
    for x, y in zip(X, Y):
      grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
      for i in range(len(weights)):
          dw[i] += grads_W[i]
          db[i] += grads_b[i]

          weights[i] -= learning_rate * dw[i]
          biases[i] -= learning_rate * db[i]
  return weights, biases