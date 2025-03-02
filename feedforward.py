import numpy as np
from activation import activation_functions, softmax

class Feedforward:
     def __init__(self, input_size, output_size, num_layers = 1, hidden_size = 4, activation = "sigmoid", weight_init = "random"):
          self.num_layers = num_layers
          self.hidden_size = hidden_size
          self.output_size = output_size
          #choose activation function & derivative
          self.activation = activation_functions[activation][0]
          self.activation_derivative = activation_functions[activation][1]
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
     
     #define forward pass
     def forward(self, x):
          z = []
          a = [x]
          for i in range(self.num_layers):
               zi = np.dot(a[i], self.weights[i]) + self.biases[i]
               z.append(zi)
               ai = self.activation(zi)
               a.append(ai)
          
          z_final = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
          z.append(z_final)
          y_cap = softmax(z_final)
          a.append(y_cap)
          return z, a
     
     #predict final output y^
     def predict(self, x):
         a = self.forward(x)[1]
         return a[-1]