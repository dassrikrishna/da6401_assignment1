import numpy as np

# cross entropy loss function L
def cross_entropy_loss(y_pred, y_true): 
  y_pred = np.clip(y_pred, 1e-15, 1-1e-15) # To prevent log(0)
  loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
  return loss

def mean_squared_loss(y_pred, y_true):
  return np.mean((y_pred - y_true) ** 2)

loss_function = {
  "mean_squared_error": mean_squared_loss,
  "cross_entropy" : cross_entropy_loss
}