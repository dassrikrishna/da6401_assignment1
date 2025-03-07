import wandb
import numpy as np
from backpropagation import *
from feedforward import forward
from loss_fun import *

# optimizer:
###################################################
# minibatch gradient descent
def minibatch_gd(X, Y, weights, biases, num_layers, learning_rate, epochs, batch_size, activation, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]
    
    # shuffle the dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]  
    X_train, Y_train = X[val_size:], Y[val_size:]

    losses = []

    for epoch in range(epochs):
        # shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
      
        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)

    return losses

##################################
# minibatch momentum based gradient descent
def minibatch_mgd(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]
    
    # suffle the dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]
    
    prev_uw = [np.zeros_like(w) for w in weights]
    prev_ub = [np.zeros_like(b) for b in biases]
    losses = []

    for epoch in range(epochs):
        # shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
      
        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)
    return losses

#####################################
# minibatch nesterov accelerated gradient descent
def minibatch_nag(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]

    # shuffle dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    prev_vw = [np.zeros_like(w) for w in weights]
    prev_vb = [np.zeros_like(b) for b in biases]
    losses = []

    for epoch in range(epochs):
        # shuffle training data
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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
        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & Accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
      
        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)

    return losses

##########################################################
def minibatch_rmsprop(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]

    # shuffle dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    v_w = [np.zeros_like(w) for w in weights]
    v_b = [np.zeros_like(b) for b in biases]
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
      
        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)

    return losses

####################
# minibatch adam gradient decent
def minibatch_adam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]

    # shuffle dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)

    return losses

################################
# minibatch nadam  gradient decent
import wandb
import numpy as np

def minibatch_nadam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun = "cross_entropy"):
    samples_size = X.shape[0]

    # shuffle dataset before splitting
    indices = np.random.permutation(samples_size)
    X, Y = X[indices], Y[indices]

    # 10% data for validation
    val_size = int(0.1 * samples_size)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

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

                # nesterov momentum in nadam
                m_w_lookahead = beta1 * m_w_cap + (1 - beta1) * (dw[j] / len(X_batch)) / (1 - beta1**(epoch + 1))
                m_b_lookahead = beta1 * m_b_cap + (1 - beta1) * (db[j] / len(X_batch)) / (1 - beta1**(epoch + 1))

                weights[j] -= (learning_rate * m_w_lookahead) / (np.sqrt(v_w_cap) + epsilon)
                biases[j] -= (learning_rate * m_b_lookahead) / (np.sqrt(v_b_cap) + epsilon)

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap)
        losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap)
        val_accuracy = compute_accuracy(Y_val, Y_val_cap)

        # wandb login
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print("epoch:", epoch + 1, "loss:", epoch_loss," & accuracy:", accuracy)

    return losses