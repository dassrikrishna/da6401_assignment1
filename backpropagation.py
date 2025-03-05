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

# minibatch momentum based gradient descent
def minibatch_mgd(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum):
  samples_size = X.shape[0]
  prev_uw = [np.zeros_like(w) for w in weights]
  prev_ub = [np.zeros_like(b) for b in biases]

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
          prev_uw[j] = momentum * prev_uw[j] + (learning_rate / len(X_batch)) * dw[j]
          prev_ub[j] = momentum * prev_ub[j] + (learning_rate / len(X_batch)) * db[j]
          weights[j] -= prev_uw[j]
          biases[j] -= prev_ub[j]

  return weights, biases

# minibatch nesterov accelerated gradient descent
def minibatch_nag(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum):
  samples_size = X.shape[0]
  prev_vw = [np.zeros_like(w) for w in weights]
  prev_vb = [np.zeros_like(b) for b in biases]

  for _ in range(epochs):
    for i in range(0, samples_size, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        dw = [np.zeros_like(w) for w in weights]
        db = [np.zeros_like(b) for b in biases]

        # Lookahead
        v_w = [momentum * pvw for pvw in prev_vw]
        v_b = [momentum * pvb for pvb in prev_vb]

        for j in range(len(weights)):
          weights[j] -= v_w[j]
          biases[j] -= v_b[j]
        
        for x, y in zip(X_batch, Y_batch):
          grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
          for j in range(len(weights)):
            dw[j] += grads_W[j]
            db[j] += grads_b[j]

        for j in range(len(weights)):
          prev_vw[j] = momentum * prev_vw[j] + (learning_rate / len(X_batch)) * dw[j]
          prev_vb[j] = momentum * prev_vb[j] + (learning_rate / len(X_batch)) * db[j]
          weights[j] -= prev_vw[j]
          biases[j] -= prev_vb[j]

  return weights, biases

# minibatch rmsprop
def minibatch_rmsprop(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon):
  samples_size = X.shape[0]
  v_w = [np.zeros_like(w) for w in weights]
  v_b = [np.zeros_like(b) for b in biases]

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
            v_w[j] = beta * v_w[j] + (1 - beta) * (dw[j] / len(X_batch))**2
            v_b[j] = beta * v_b[j] + (1 - beta) * (db[j] / len(X_batch))**2

            weights[j] -= (learning_rate * (dw[j] / len(X_batch))) / (np.sqrt(v_w[j]) + epsilon)
            biases[j] -= (learning_rate * (db[j] / len(X_batch))) / (np.sqrt(v_b[j]) + epsilon)

  return weights, biases

# minibatch adam gradient decent
def minibatch_adam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon):
  samples_size = X.shape[0]
  m_w = [np.zeros_like(w) for w in weights]
  v_w = [np.zeros_like(w) for w in weights]
  m_b = [np.zeros_like(b) for b in biases]
  v_b = [np.zeros_like(b) for b in biases]

  for epoch in range(epochs):
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
            m_w[j] = beta1 * m_w[j] + (1 - beta1) * (dw[j] / len(X_batch))
            v_w[j] = beta2 * v_w[j] + (1 - beta2) * ((dw[j] / len(X_batch))**2)
            m_b[j] = beta1 * m_b[j] + (1 - beta1) * (db[j] / len(X_batch))
            v_b[j] = beta2 * v_b[j] + (1 - beta2) * ((db[j] / len(X_batch))**2)

            m_w_cap = m_w[j] / (1 - beta1**(epoch + 1))
            v_w_cap = v_w[j] / (1 - beta2**(epoch + 1))
            m_b_cap = m_b[j] / (1 - beta1**(epoch + 1))
            v_b_cap = v_b[j] / (1 - beta2**(epoch + 1))

            weights[j] -= (learning_rate * m_w_cap) / (np.sqrt(v_w_cap) + epsilon)
            biases[j] -= (learning_rate * m_b_cap) / (np.sqrt(v_b_cap) + epsilon)

  return weights, biases

# minibatch nadam  gradient decent
def minibatch_nadam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon):
    samples_size = X.shape[0]
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
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
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * (dw[j] / len(X_batch))
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * ((dw[j] / len(X_batch))**2)
                m_b[j] = beta1 * m_b[j] + (1 - beta1) * (db[j] / len(X_batch))
                v_b[j] = beta2 * v_b[j] + (1 - beta2) * ((db[j] / len(X_batch))**2)

                m_w_cap = m_w[j] / (1 - beta1**(epoch + 1))
                v_w_cap = v_w[j] / (1 - beta2**(epoch + 1))
                m_b_cap = m_b[j] / (1 - beta1**(epoch + 1))
                v_b_cap = v_b[j] / (1 - beta2**(epoch + 1))

                # nesterov momentum
                m_w_lookahead = beta1 * m_w_cap + (1 - beta1) * (dw[j] / len(X_batch)) / (1 - beta1**(epoch + 1))

                weights[j] -= (learning_rate * m_w_lookahead) / (np.sqrt(v_w_cap) + epsilon)
                biases[j] -= (learning_rate * m_b_cap) / (np.sqrt(v_b_cap) + epsilon)

    return weights, biases