import wandb
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from reshape_data import transform, one_hot
from feedforward import initialize_weights
from backpropagation import *
from optimizer import *
# wandb login
wandb.login()
wandb.init(project = "MA24M025_DA6401-Assignment-1", mode = "online")

x_train, y_train = transform(x_train), one_hot(y_train)
x_test, y_test = transform(x_test), one_hot(y_test)

input_size = 784
hidden_size = 4
output_size = 10
num_layers = 1
weight_init = "random"
activation = "sigmoid"
learning_rate = 0.01
momentum = 0.5
epochs = 10
batch_size = 32
beta = 0.5
beta1 = 0.5
beta2 = 0.5
epsilon = 1e-8
loss_fun = "cross_entropy"

"""weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_gd(x_train, y_train, weights, biases, num_layers, learning_rate, epochs, batch_size, activation)"""

"""weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_mgd(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum)"""

"""weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_nag(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum)"""

"""weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_rmsprop(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon)"""

"""weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_adam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon)"""

weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
minibatch_nadam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon, loss_fun)

wandb.finish()