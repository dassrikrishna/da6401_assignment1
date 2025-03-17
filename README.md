# DA6401 Assignment-1
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
- A **Gradescope submission** linking to the project.
#### Detailed Weights & Biases Report for My Project: [HERE](https://wandb.ai/ma24m025-indian-institute-of-technology-madras/MA24M025_DA6401_Project-1/reports/MA24M025_DA6401-Assignment-1-Report--VmlldzoxMTY5MDE0NQ)

## **Table of Contents**
- [Installation](#installation)
- [Project Structure](#project-structure)
## **Installation**
To set up the project, clone the repository and install the required dependencies:
```bash
git clone https://github.com/dassrikrishna/da6401_assignment1.git
cd da6401_assignment1
pip install -r requirements.txt
```
## **Project Structure**
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
  - Runs an optimizer on a sample input.
  - Outputs `epoch`, `epoch_loss`, `accuracy`, and updated weights & biases.

- `question4n7.py`
  - Performs hyperparameter tuning using Bayesian optimization.
  - Defines `sweep_config` with tunable parameters:
    - Number of epochs
    - Number of layers
    - Hidden layer size
    - Weight decay (L2 regularization)
    - Learning rate
    - Optimizer
    - Batch size
    - Weight initialization method
    - Activation function
  - Runs sweeps and evaluates test accuracy and confusion matrix for each experiment.
    

