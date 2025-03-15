import numpy as np

# cross entropy loss function L
def cross_entropy_loss(y_pred, y_true, weights, biases, weight_decay): 
  y_pred = np.clip(y_pred, 1e-15, 1-1e-15) # To prevent log(0)
  loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

  l2_reg = (weight_decay / 2) * (sum(np.linalg.norm(w, 'fro') ** 2 for w in weights) + sum(np.linalg.norm(b) ** 2 for b in biases))
    
  return loss + l2_reg

def mean_squared_loss(y_pred, y_true, weights, biases, weight_decay):
  l2_reg = (weight_decay / 2) * (sum(np.linalg.norm(w, 'fro') ** 2 for w in weights) + sum(np.linalg.norm(b) ** 2 for b in biases))

  return np.mean(0.5*(y_pred - y_true) ** 2) + l2_reg

loss_function = {
  "mean_squared_error": mean_squared_loss,
  "cross_entropy" : cross_entropy_loss
}
