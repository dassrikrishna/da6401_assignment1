import numpy as np
from activation import activation_functions, softmax

class Feedforward:
     def __init__(self, input_size, num_layers, hidden_size, output_size, activation = "sigmoid", weight_init = "random"):
          self.num_layers = num_layers
          self.hidden_size = hidden_size
          self.output_size = output_size
          #choose activation function 
          self.activation = activation_functions[activation][0]
          #intialize weights and biases
          self.weights, self.biases = self.initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)
     
     
     def initialize_weights(self, input_size, hidden_size, output_size, num_layers, weight_init):
        weights = []
        biases = []