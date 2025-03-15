import argparse
import wandb
from keras.datasets import fashion_mnist, mnist
from reshape_data import transform, one_hot
from feedforward import initialize_weights
from optimizer import *


def parse_args():
    parser = argparse.ArgumentParser(description = 'Train a Feedforward Neural Network')
    parser.add_argument('-wp', '--wandb_project', default = 'DA6401train-Project', help = 'Project name for Weights & Biases')
    parser.add_argument('-we', '--wandb_entity', default = 'ma24m025-indian-institute-of-technology-madras', help = 'Entity for Weights & Biases')
    parser.add_argument('-d', '--dataset', choices = ['fashion_mnist', 'mnist'], default = 'fashion_mnist', help = 'Dataset selection')
    parser.add_argument('-e', '--epochs', type = int, default = 5, help = 'Number of epochs')
    parser.add_argument('-b', '--batch_size', type = int, default = 4, help = 'Batch size')
    parser.add_argument('-l', '--loss_fun', choices = ['mean_squared_error', 'cross_entropy'], default = 'cross_entropy', help = 'Loss function')
    parser.add_argument('-o', '--optimizer', choices = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'], default = 'sgd', help = 'Optimizer')
    parser.add_argument('-lr', '--learning_rate', type = float, default = 0.1, help = 'Learning rate')
    parser.add_argument('-m', '--momentum', type = float, default = 0.5, help = 'Momentum for momentum and nesterov optimizers')
    parser.add_argument('-beta', '--beta', type = float, default = 0.5, help = 'Beta for rmsprop')
    parser.add_argument('-beta1', '--beta1', type = float, default = 0.5, help = 'Beta1 for adam and nadam')
    parser.add_argument('-beta2', '--beta2', type = float, default = 0.5, help = 'Beta2 for adam and nadam')
    parser.add_argument('-eps', '--epsilon', type = float, default = 1e-6, help = 'Epsilon for optimizers')
    parser.add_argument('-w_i', '--weight_init', choices = ['random', 'Xavier'], default = 'random', help = 'Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type = int, default = 1, help = 'Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type = int, default = 4, help = 'Hidden layer size')
    parser.add_argument('-a', '--activation', choices = ['identity', 'sigmoid', 'tanh', 'ReLU'], default = 'sigmoid', help = 'Activation function')
    parser.add_argument('-w_d', '--weight_decay',type = float, default = 0.0, help = "weight decay for L2 regurlarization")
    return parser.parse_args()


def train():
    args = parse_args()
    wandb.init(project = args.wandb_project, entity = args.wandb_entity, config = vars(args))
    config = wandb.config

    if config.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = transform(x_train), one_hot(y_train)
    x_test, y_test = transform(x_test), one_hot(y_test)

    input_size, output_size = 784, 10
    weights, biases = initialize_weights(input_size, config.hidden_size, output_size, config.num_layers, config.weight_init)

    if config.optimizer == "sgd":
        minibatch_gd(x_train, y_train, weights, biases, config.num_layers, config.learning_rate, config.epochs, config.batch_size, config.activation, config.loss_fun, config.weight_decay)
    elif config.optimizer == "momentum":
        minibatch_mgd(x_train, y_train, weights, biases, config.num_layers, config.activation, config.learning_rate, config.epochs, config.batch_size, config.momentum, config.loss_fun, config.weight_decay)
    elif config.optimizer == "nesterov":
        minibatch_nag(x_train, y_train, weights, biases, config.num_layers, config.activation, config.learning_rate, config.epochs, config.batch_size, config.momentum, config.loss_fun, config.weight_decay)
    elif config.optimizer == "rmsprop":
        minibatch_rmsprop(x_train, y_train, weights, biases, config.num_layers, config.activation, config.learning_rate, config.epochs, config.batch_size, config.beta, config.epsilon, config.loss_fun, config.weight_decay)
    elif config.optimizer == "adam":
        minibatch_adam(x_train, y_train, weights, biases, config.num_layers, config.activation, config.learning_rate, config.epochs, config.batch_size, config.beta1, config.beta2, config.epsilon, config.loss_fun, config.weight_decay)
    elif config.optimizer == "nadam":
        minibatch_nadam(x_train, y_train, weights, biases, config.num_layers, config.activation, config.learning_rate, config.epochs, config.batch_size, config.beta1, config.beta2, config.epsilon, config.loss_fun, config.weight_decay)

    wandb.finish()


if __name__ == "__main__":
    train()