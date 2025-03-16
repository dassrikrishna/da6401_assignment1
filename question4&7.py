import wandb
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from reshape_data import transform, one_hot
from feedforward import initialize_weights
from optimizer import *
from taccuracy_confusion import *

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
    weight_decay = config.weight_decay

    momentum = 0.5
    beta = 0.5
    epsilon = 1e-8
    beta1 = 0.5
    beta2 = 0.5
    loss_fun = "cross_entropy"
    #loss_fun = "mean_squared_error"

    # run name
    run_name = f"ep-{epochs}_hl-{num_layers}_hs-{hidden_size}_wi-{weight_init}_bs-{batch_size}_lr{learning_rate}_ac-{activation}_op-{optimizer}_wd-{weight_decay}"
    wandb.run.name = run_name

    # initialize weights
    weights, biases = initialize_weights(input_size, hidden_size, output_size, num_layers, weight_init)

    # Choose optimizer
    if optimizer == "sgd":
        weights, biases = minibatch_gd(x_train, y_train, weights, biases, num_layers, learning_rate, epochs, batch_size, activation,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
    elif optimizer == "momentum":
        weights, biases = minibatch_mgd(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
    elif optimizer == "nesterov":
        weights, biases = minibatch_nag(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, momentum,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
    elif optimizer == "rmsprop":
        weights, biases = minibatch_rmsprop(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta, epsilon,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
    elif optimizer == "adam":
        weights, biases = minibatch_adam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
    elif optimizer == "nadam":
        weights, biases = minibatch_nadam(x_train, y_train, weights, biases, num_layers, activation, learning_rate, epochs, batch_size, beta1, beta2, epsilon,  loss_fun, weight_decay)
        cal_taccu_confu_mat(x_test, y_test, num_layers, weights, biases, activation)
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
        "weight_decay": {"values": [0, 0.0005, 0.5]},
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
    sweep_id = wandb.sweep(sweep_config, project = "CrossEntropy67")
    wandb.agent(sweep_id, function = train, count = 67)  # runs 67 experiments
