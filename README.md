# DA6401 Assignment-1
## DEEP LEARNING
#### ```SRIKRISHNA DAS (MA24M025)```
#### `M.Tech (Industrial Mathematics and Scientific Computing)` `IIT Madras`
 

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
