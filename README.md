# DA6401 Assignment-1
#### Detailed Weights & Biases Report for My Project: [Click Here](https://wandb.ai/ma24m025-indian-institute-of-technology-madras/MA24M025_DA6401_Project-1/reports/MA24M025_DA6401-Assignment-1-Report--VmlldzoxMTY5MDE0NQ)
#### Github Link: [Click Here](https://github.com/dassrikrishna/da6401_assignment1)
## DEEP LEARNING
#### ```SRIKRISHNA DAS (MA24M025)```
#### `M.Tech (Industrial Mathematics and Scientific Computing) IIT Madras`
 

## [Problem Statement](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-1--VmlldzoxMTQ2NDQwNw)

In this assignment, you need to implement a **feedforward neural network** and write the **backpropagation** code for training the network. The model will be trained and tested on the **Fashion-MNIST** dataset. Given an input image (**28 × 28 = 784 pixels**), the network should classify it into **one of 10 classes**.  

You must implement the neural network using **NumPy** for all matrix and vector operations. The use of **automatic differentiation libraries** such as TensorFlow, Keras, or PyTorch is **not allowed**.  

The model should support:  
- **Flexible architecture**: The number of **hidden layers** and **neurons per layer** should be configurable.  
- **Multiple activation functions**: Identity, Sigmoid, Tanh, and ReLU for hidden layers, with **Softmax** for the output layer.  
- **Optimization techniques**: Implement **SGD, Momentum, NAG, RMSprop, Adam, and Nadam** for training.  
- **Loss functions**: Implement **cross-entropy loss** and **mean squared error loss**, both with **L2 regularization**.  

Additionally, you need to track and log your experiments using **Weights & Biases (wandb)**, conduct **hyperparameter tuning** using **wandb sweeps**, and analyze the model's performance using **test accuracy and a confusion matrix**.  

A **```train.py```** script must be implemented to allow command-line configuration of hyperparameters and training options.  

Your submission should include:  
- A **GitHub repository** with all required code.  
- A **wandb report** summarizing experiments and findings.  

#### Detailed Weights & Biases Report for My Project: [Click Here](https://wandb.ai/ma24m025-indian-institute-of-technology-madras/MA24M025_DA6401_Project-1/reports/MA24M025_DA6401-Assignment-1-Report--VmlldzoxMTY5MDE0NQ)

## **Table of Contents**
- [Installation](#installation)
- [Project Structure](#project-structure)
  
   - [Data Processing](#data-processing)
   - [Neural Network Components](#neural-network-components)
   - [Optimization and Performance Evaluation](#optimization-and-performance-evaluation)
   - [Evaluation and Experimentation](#evaluation-and-experimentation)
     
- [Training Script (train.py)](#training-script-trainpy)
  
  - [Usage Examples](#usage-examples)
  - [Arguments to be supported](#arguments-to-be-supported)
  
## **Installation**
To set up the project, clone the repository and install the required dependencies:
```bash
git clone https://github.com/dassrikrishna/da6401_assignment1.git
cd da6401_assignment1
pip install -r requirements.txt
```
## **Project Structure**
```bash
|-- train.py                         # Main script to train the neural network
|-- reshape_data.py                  # Preprocessing: normalize, reshape, and one-hot encode
|-- activation.py                    # Activation functions and their derivatives
|-- loss_fun.py                      # Cross-entropy loss and MSE with L2 regularization
|-- accuracy_confusion.py            # Accuracy computation and confusion matrix
|-- fashion_mnist_overview.py        # Dataset loading and visualization
|-- feedforward.py                   # Forward propagation with Xavier & Random Normal init
|-- question2.py                     # Outputs probability distribution for a sample image
|-- backpropagation.py               # Computes gradients for weight updates
|-- optimizer.py                     # Implements multiple optimizers
|-- question3.py                     # Runs an optimizer and outputs updated weights
|-- question4n7.py                   # Hyperparameter tuning & test accuracy tracking
|-- README.md                        # Project documentation
|-- requirements.txt                 # Dependencies  
```
### **Data Processing**
- `reshape_data.py`
  - Normalizes images to [0,1] range.
  - Reshapes 28×28 images to a 1D vector (784,).
  - Define one-hot encoding for output.
- **Q1.**`fashion_mnist_overview.py`
  - Loads the Fashion-MNIST dataset using `keras.datasets.fashion_mnist`.
  - Displays sample images for each class.
  - Logs images to Weights & Biases.
### **Neural Network Components**
- `activation.py`
  - Implements identity, sigmoid, tanh, and ReLU activation functions and their derivatives.
  - Defines softmax activation for the output layer.

- `loss_fun.py`
  - Implements cross-entropy and mean squared error (MSE) loss.
  - Supports L2 regularization (weight decay).

- `feedforward.py`
  - Initializes weights and biases using random normal and Xavier initialization.
  - Implements the forward pass, returning pre-activation and activation values.
  - Computes predicted outputs (`y_cap`).

- `backpropagation.py`
  - Uses activation functions from `activation.py` and the forward pass from `feedforward.py`.
  - Computes gradients $\frac{\partial L}{\partial W}, \frac{\partial L}{\partial B}$ for each layer.

### **Optimization and Performance Evaluation**
- `optimizer.py`
  - Implements multiple optimization algorithms:
    - Gradient Descent (GD)
    - Momentum GD
    - Nesterov Accelerated Gradient (NAG)
    - RMSprop
    - Adam
    - Nadam
  - Supports mini-batch training (90% training, 10% validation).
  - Logs training progress (`epoch_loss`, `train_accuracy`, `val_loss`, `val_accuracy`) to wandb.

- `taccuracy_confusion.py`
  - Computes accuracy (`compute_accuracy`).
  - Evaluates test accuracy and generates confusion matrix (`cal_taccu_confu_mat`).

### **Evaluation and Experimentation**
- `question2.py`
  - Takes a sample image as input.
  - Outputs probability distribution for each class using `feedforward.py`.

- `question3.py`
  - Data procesing (change shape, one-hot-encoding) using `reshape_data.py`. 
  - Runs an optimizer on a sample input.
  - Implemented `compute_grads` in `backpropagation.py` to calculate gradients of weights and biases using backpropagation.
  - Used 1compute_grads` in `optimizer.py` to define various optimizers.
  - Incorporated the forward pass in each optimizer for weight updates.
  - Outputs `epoch`, `epoch_loss`, `accuracy`, and updated weights & biases.

- `question4n7.py`
  - Performs hyperparameter tuning using Bayesian optimization.
  - Defines `sweep_config` with tunable parameters:
    - Number of epochs: 5, 10
    - Number of layers: 3, 4, 5
    - Hidden layer size: 32, 64, 128
    - Weight decay (L2 regularization): 0, 0.0005, 0.5
    - Learning rate: 0.001, 0.0001
    - Optimizer: "sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"
    - Batch size: 16, 32, 64
    - Weight initialization method: "random", "Xavier"
    - Activation function: "sigmoid", "tanh", "ReLU"
  - Runs sweeps and evaluates test accuracy and confusion matrix for each experiment.
    
## **Training Script (train.py)**  
- Implements the **main training pipeline** for the feedforward neural network.  
- Supports **command-line configuration** using `argparse` for flexible training options.  
- Integrates with **Weights & Biases (`wandb`)** for logging hyperparameters, loss, and accuracy.  

### **Key Features:**  
- **Command-Line Interface (CLI)**  
  - Configures hyperparameters dynamically via command-line arguments.  
  - Supports dataset selection, training settings, network architecture, optimization methods, and regularization parameters.  

- **Training with Optimizers**  
  - Implements different optimizers from `optimizer.py`:  
    - **SGD, Momentum, NAG, RMSprop, Adam, Nadam**  
  - Calls the corresponding optimizer function to perform forward pass, gradient computation (`compute_grads`), and weight updates.  

- **Evaluation & Logging**  
  - Computes **test accuracy** and generates a **confusion matrix** using `cal_taccu_confu_mat` from `taccuracy_confusion.py`.  
  - Logs **epoch_loss, train_accuracy, val_loss, val_accuracy** to **Weights & Biases (`wandb`)**.  

### **Usage Examples**  
```bash
python train.py --wandb_entity ma24m025-indian-institute-of-technology-madras --wandb_project DA6401train-Project
```

```bash
python train.py --wandb_entity ma24m025-indian-institute-of-technology-madras --wandb_project DA6401train-Project --dataset mnist --epochs 10 --batch_size 64 --optimizer nadam --learning_rate 0.001 --num_layers 3 --hidden_size 128 --activation tanh --weight_init Xavier --weight_decay 0.0 --beta1 0.9 --beta2 0.99
```
### **Supported Arguments**

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | DA6401train-Project | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | ma24m025-indian-institute-of-technology-madras  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 64 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | ReLU | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>
