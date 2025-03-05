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

# optimization:
# minibatch gradient descent
# stochastic if batch_size = 1

def minibatch_gd(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size):

  samples_size = X.shape[0]
  for _ in range(epochs):

    for i in range(0, samples_size, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        dw = [np.zeros_like(w) for w in weights]
        db = [np.zeros_like(b) for b in biases]
        
        for x, y in zip(X_batch, Y_batch):
          grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
          for j in range(len(weights)):
            dw[j] += grads_W[j]
            db[j] += grads_b[j]
        
        for j in range(len(weights)):
            weights[j] -= (learning_rate / len(X_batch)) * dw[j]  
            biases[j] -= (learning_rate / len(X_batch)) * db[j]    

  return weights, biases