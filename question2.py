# question 2:
from feedforward import forward,initialize_weights
from reshape_data import transform
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = transform(x_train)

input_size = 784
hidden_size = 4
output_size = 10
num_layers = 1
weight_init = "random"
activation = "sigmoid"

weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)

# input sample image x_train[0]
_, a = forward(x_train[0], num_layers, weights, biases, activation)

# output probability distribution
print(a[-1])