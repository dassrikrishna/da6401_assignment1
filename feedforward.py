import numpy as np
from activation import activation_functions, softmax

# initialize the weight by random normal or Xavier
def initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init):
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

# define forward pass
def forward(x, num_layers, weights, biases, activation):
  z = []
  a = [x]
  activation_name = activation_functions[activation][0]
  for i in range(num_layers):
        zi = np.dot(a[i], weights[i]) + biases[i]
        z.append(zi)
        ai = activation_name(zi)
        a.append(ai)

  z_final = np.dot(a[-1], weights[-1]) + biases[-1]
  z.append(z_final)
  y_cap = softmax(z_final)
  a.append(y_cap)
  return z, a