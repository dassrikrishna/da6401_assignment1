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
     
     #initialize weights "random normal" or "Xavier"
     def initialize_weights(self, input_size, hidden_size, output_size, num_layers, weight_init):
        weights = []
        biases = []

        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            if weight_init == "random":
                weight_matrix = np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i + 1]))
            elif weight_init == "Xavier":
                sigma = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                weight_matrix = np.random.uniform(-sigma, sigma, (layer_sizes[i], layer_sizes[i + 1]))
            
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            weights.append(weight_matrix)
            biases.append(bias_vector)
        
        return weights, biases