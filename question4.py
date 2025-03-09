import wandb
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from reshape_data import transform, one_hot
from feedforward import initialize_weights
from optimizer import *

x_train, y_train = transform(x_train), one_hot(y_train)
x_test, y_test = transform(x_test), one_hot(y_test)

def train():
    wandb.init()
    config = wandb.config
    
    # tuning hyperparameter
    input_size = 784
    output_size = 10

    num_layers = config.num_layers
    hidden_size = config.hidden_size
    weight_init = config.weight_init
    activation = config.activation
    learning_rate = config.learning_rate
    epochs = config.epochs
    batch_size = config.batch_size
    optimizer = config.optimizer

    momentum = 0.5
    beta = 0.5
    epsilon = 1e-8
    beta1 = 0.5
    beta2 = 0.5

    # initialize weights
    weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)

    # Choose optimizer
    if optimizer == "sgd":
        minibatch_gd(x_train, y_train, weights, biases, num_layers, learning_rate, epochs, batch_size, activation)
    elif optimizer == "momentum":
        minibatch_mgd(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum)
    elif optimizer == "nesterov":
        minibatch_nag(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum)
    elif optimizer == "rmsprop":
        minibatch_rmsprop(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon)
    elif optimizer == "adam":
        minibatch_adam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon)
    elif optimizer == "nadam":
        minibatch_nadam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon)
    
    wandb.finish()

# load sweep_config 
sweep_config = {
    "program": "question4.py",
    "method": "bayes",  # bayesian Optimization for efficiency
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        #"weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [0.001, 0.0001]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]}
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
}

# Initialize sweep and run directly
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project = "MA24M025_DA6401_Project-1")
    wandb.agent(sweep_id, function = train, count = 50)  # runs 50 experiments
