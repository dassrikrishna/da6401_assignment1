import numpy as np
from backpropagation import compute_grads
from feedforward import forward
from loss_fun import *
# optimizer:
# minibatch gradient descent
# stochastic if batch_size = 1

# minibatch gradient descent
def minibatch_gd(X, Y, weights, biases, num_layers, learning_rate, epochs, batch_size, activation, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  losses = []
  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]
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
      
    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses

# minibatch momentum based gradient descent
def minibatch_mgd(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  prev_uw = [np.zeros_like(w) for w in weights]
  prev_ub = [np.zeros_like(b) for b in biases]
  losses = []
  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]
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

    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses

# minibatch nesterov accelerated gradient descent
def minibatch_nag(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  prev_vw = [np.zeros_like(w) for w in weights]
  prev_vb = [np.zeros_like(b) for b in biases]
  losses = []

  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]
    for i in range(0, samples_size, batch_size):
      X_batch = X[i:i + batch_size]
      Y_batch = Y[i:i + batch_size]

      dw = [np.zeros_like(w) for w in weights]
      db = [np.zeros_like(b) for b in biases]

      # lookahead
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

    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses

# minibatch rmsprop
def minibatch_rmsprop(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  v_w = [np.zeros_like(w) for w in weights]
  v_b = [np.zeros_like(b) for b in biases]
  losses = []
  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]
    for i in range(0, samples_size, batch_size):
      X_batch = X[i:i + batch_size]
      Y_batch = Y[i:i + batch_size]

      dw = [np.zeros_like(w) for w in weights]
      db = [np.zeros_like(b) for b in biases]
      
      for x, y in zip(X_batch, Y_batch):
        grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
        for j in range(len(weights)):
          dw[j] = grads_W[j]
          db[j] = grads_b[j]

      for j in range(len(weights)):
          v_w[j] = beta * v_w[j] + (1 - beta) * (dw[j] / len(X_batch))**2
          v_b[j] = beta * v_b[j] + (1 - beta) * (db[j] / len(X_batch))**2

          weights[j] -= (learning_rate * (dw[j] / len(X_batch))) / (np.sqrt(v_w[j]) + epsilon)
          biases[j] -= (learning_rate * (db[j] / len(X_batch))) / (np.sqrt(v_b[j]) + epsilon)

    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses

# minibatch adam gradient decent
def minibatch_adam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  m_w = [np.zeros_like(w) for w in weights]
  v_w = [np.zeros_like(w) for w in weights]
  m_b = [np.zeros_like(b) for b in biases]
  v_b = [np.zeros_like(b) for b in biases]
  losses = []
  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]    
    for i in range(0, samples_size, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        dw = [np.zeros_like(w) for w in weights]
        db = [np.zeros_like(b) for b in biases]
        
        for x, y in zip(X_batch, Y_batch):
          grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
          for j in range(len(weights)):
            dw[j] = grads_W[j]
            db[j] = grads_b[j]
        
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

    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses

# minibatch nadam  gradient decent
def minibatch_nadam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun = "cross_entropy"):
  samples_size = X.shape[0]
  m_w = [np.zeros_like(w) for w in weights]
  v_w = [np.zeros_like(w) for w in weights]
  m_b = [np.zeros_like(b) for b in biases]
  v_b = [np.zeros_like(b) for b in biases]
  losses =[]
  for epoch in range(epochs):
    indices = np.random.permutation(samples_size)
    X = X[indices]
    Y = Y[indices]    
    for i in range(0, samples_size, batch_size):
      X_batch = X[i:i + batch_size]
      Y_batch = Y[i:i + batch_size]

      dw = [np.zeros_like(w) for w in weights]
      db = [np.zeros_like(b) for b in biases]

      for x, y in zip(X_batch, Y_batch):
        grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation)
        for j in range(len(weights)):
          dw[j] = grads_W[j]
          db[j] = grads_b[j]

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

    Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X]).squeeze()  
    epoch_loss = loss_function[loss_fun](Y, Y_cap)
    losses.append(epoch_loss)     
    # print epoch and epoch_loss at the end of each epoch
    print("epoch:", epoch + 1, "loss:", epoch_loss)

  return losses