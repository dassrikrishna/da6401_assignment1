import wandb
import numpy as np
from backpropagation import *
from feedforward import forward
from loss_fun import *
from taccuracy_confusion import compute_accuracy

# optimizer:
###################################################
# minibatch gradient descent
def minibatch_gd(X, Y, weights, biases, num_layers, learning_rate, epochs, batch_size, activation, loss_fun, weight_decay):
    samples_size = X.shape[0]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]

        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]

            for x, y in zip(X_batch, Y_batch):
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)
                for j in range(len(weights)):
                    dw[j] += grads_W[j]
                    db[j] += grads_b[j]

            for j in range(len(weights)):
                weights[j] -= (learning_rate / len(X_batch)) * dw[j]
                biases[j] -= (learning_rate / len(X_batch)) * db[j]

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases

##################################
# minibatch momentum based gradient descent
def minibatch_mgd(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun, weight_decay):
    samples_size = X.shape[0]
    
    prev_uw = [np.zeros_like(w) for w in weights]
    prev_ub = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]
        
        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]

            for x, y in zip(X_batch, Y_batch):
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)
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
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases

#####################################
# minibatch nesterov accelerated gradient descent
def minibatch_nag(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum, loss_fun, weight_decay):
    samples_size = X.shape[0]

    prev_vw = [np.zeros_like(w) for w in weights]
    prev_vb = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]
        
        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

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
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)
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
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & Accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases

##########################################################
def minibatch_rmsprop(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon, loss_fun, weight_decay):
    samples_size = X.shape[0]

    v_w = [np.zeros_like(w) for w in weights]
    v_b = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]
        
        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]

            for x, y in zip(X_batch, Y_batch):
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)
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
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases

####################

# minibatch adam gradient decent
def minibatch_adam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun, weight_decay):
    samples_size = X.shape[0]
    t = 0 # step

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]
        
        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]

            for x, y in zip(X_batch, Y_batch):
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)

                dw = grads_W
                db = grads_b
            t +=1 # increment step
            for j in range(len(weights)):
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * (dw[j] / len(X_batch))
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * ((dw[j] / len(X_batch))**2)
                m_b[j] = beta1 * m_b[j] + (1 - beta1) * (db[j] / len(X_batch))
                v_b[j] = beta2 * v_b[j] + (1 - beta2) * ((db[j] / len(X_batch))**2)

                m_w_cap = m_w[j] / (1 - beta1**t)
                v_w_cap = v_w[j] / (1 - beta2**t)
                m_b_cap = m_b[j] / (1 - beta1**t)
                v_b_cap = v_b[j] / (1 - beta2**t)

                weights[j] -= (learning_rate * m_w_cap) / (np.sqrt(v_w_cap) + epsilon)
                biases[j] -= (learning_rate * m_b_cap) / (np.sqrt(v_b_cap) + epsilon)

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases

################################
# minibatch nadam  gradient decent
def minibatch_nadam(X, Y, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun, weight_decay):
    samples_size = X.shape[0]
    t = 0 # step

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    for epoch in range(epochs):
        # shuffle the dataset before splitting
        indices = np.random.permutation(samples_size)
        X, Y = X[indices], Y[indices]

        # 10% data for validation
        val_size = int(0.1 * samples_size)
        X_val, Y_val = X[:val_size], Y[:val_size]  
        X_train, Y_train = X[val_size:], Y[val_size:]
        
        # shuffle training data
        train_indices = np.random.permutation(X_train.shape[0])
        X_train, Y_train = X_train[train_indices], Y_train[train_indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]

            for x, y in zip(X_batch, Y_batch):
                grads_W, grads_b = compute_grads(x, y, num_layers, weights, biases, activation, weight_decay)
                
                dw = grads_W
                db = grads_b

            t += 1 # increment step
            for j in range(len(weights)):
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * (dw[j] / len(X_batch))
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * ((dw[j] / len(X_batch))**2)
                m_b[j] = beta1 * m_b[j] + (1 - beta1) * (db[j] / len(X_batch))
                v_b[j] = beta2 * v_b[j] + (1 - beta2) * ((db[j] / len(X_batch))**2)

                m_w_cap = m_w[j] / (1 - beta1**t)
                v_w_cap = v_w[j] / (1 - beta2**t)
                m_b_cap = m_b[j] / (1 - beta1**t)
                v_b_cap = v_b[j] / (1 - beta2**t)

                # nesterov momentum in nadam
                m_w_lookahead = beta1 * m_w_cap + (1 - beta1) * (dw[j] / len(X_batch)) / (1 - beta1**t)
                m_b_lookahead = beta1 * m_b_cap + (1 - beta1) * (db[j] / len(X_batch)) / (1 - beta1**t)

                weights[j] -= (learning_rate * m_w_lookahead) / (np.sqrt(v_w_cap) + epsilon)
                biases[j] -= (learning_rate * m_b_lookahead) / (np.sqrt(v_b_cap) + epsilon)

        # training loss & accuracy
        Y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_train]).squeeze()
        epoch_loss = loss_function[loss_fun](Y_train, Y_cap, weights, biases, weight_decay)
        #losses.append(epoch_loss)
        accuracy = compute_accuracy(Y_train, Y_cap)

        # validation loss & accuracy
        Y_val_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in X_val]).squeeze()
        val_loss = loss_function[loss_fun](Y_val, Y_val_cap, weights, biases, weight_decay)
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
    return weights, biases
