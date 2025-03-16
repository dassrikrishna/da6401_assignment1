import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from feedforward import *

def compute_accuracy(Y_true, Y_pred):
  true_values = np.argmax(Y_true, axis=1)
  pred_values = np.argmax(Y_pred, axis=1)  
  return np.mean(true_values == pred_values)

# compute confusion matrix
def cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation):
    
    y_cap = np.array([forward(x, num_layers, weights, biases, activation)[1][-1] for x in x_test]).squeeze()
    test_accuracy = compute_accuracy(y_test, y_cap)

    y_test_pred = np.argmax(y_cap, axis = 1)
    Y_test_true = np.argmax(y_test, axis = 1)
    conf_matrix = confusion_matrix(Y_test_true, y_test_pred)

    # plot confusion matrix
    plt.figure(figsize=(8, 8))

    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    print("Test-Accuracy",test_accuracy)
    #  log accuracy and confusion matrix to wandb
    wandb.log({"Test-Accuracy": test_accuracy})
    wandb.log({"Confusion-Matrix": wandb.Image(plt)})