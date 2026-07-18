# Day 7 - Neural Networks with PyTorch

## Overview

This project explores the fundamentals of building and training neural networks using PyTorch's `torch.nn` package. It demonstrates how neural networks are structured, how forward and backward propagation work, how loss is computed, and how model parameters are updated during training.

The implementation is based on a Convolutional Neural Network (CNN) architecture similar to **LeNet**, designed for image classification tasks.

---

## Key Concepts Covered

### Neural Network Architecture

* Creating custom models using `nn.Module`
* Defining layers such as:

  * Convolutional Layers (`nn.Conv2d`)
  * Fully Connected Layers (`nn.Linear`)
  * Activation Functions (ReLU)
  * Max Pooling Layers

### Forward Propagation

* Processing input data through multiple layers
* Feature extraction using convolution operations
* Flattening feature maps for classification

### Automatic Differentiation

* Understanding PyTorch's `autograd` engine
* Tracking computational graphs
* Gradient computation using `backward()`

### Loss Functions

* Calculating prediction error using:

  * Mean Squared Error (`nn.MSELoss`)
* Comparing model outputs with target values

### Backpropagation

* Computing gradients automatically
* Exploring gradient flow through the network
* Understanding parameter updates

### Optimization

* Manual gradient-based weight updates
* Using PyTorch optimizers
* Training with Stochastic Gradient Descent (SGD)

---

## Network Architecture

Input (32×32 Grayscale Image)

↓ Conv2D (1 → 6)

↓ ReLU

↓ MaxPool

↓ Conv2D (6 → 16)

↓ ReLU

↓ MaxPool

↓ Flatten

↓ Linear (400 → 120)

↓ ReLU

↓ Linear (120 → 84)

↓ ReLU

↓ Linear (84 → 10)

↓ Output

---

## Technologies Used

* Python
* PyTorch
* Torch.nn
* Torch.optim
* Autograd

---

## Learning Outcomes

Through this project, I learned:

* How neural networks are implemented in PyTorch
* The role of convolutional and fully connected layers
* How forward and backward propagation work
* How loss functions guide model learning
* How gradients are computed and used for optimization
* The importance of optimizers and learning rates in training

# Neural Networks with PyTorch

## Overview

This project explores the fundamentals of building and training neural networks using PyTorch. It introduces the `torch.nn` package, automatic differentiation with `autograd`, loss functions, backpropagation, and optimization techniques used to train deep learning models.

A Convolutional Neural Network (CNN) inspired by the classic **LeNet architecture** is implemented to demonstrate the complete neural network workflow.

---

## Learning Objectives

By completing this project, you will learn:

* How to build neural networks using `nn.Module`
* How convolutional and fully connected layers work
* Forward propagation through a neural network
* Automatic gradient computation with `autograd`
* Loss calculation using PyTorch loss functions
* Backpropagation and gradient updates
* Training models using optimizers such as SGD

---

## Project Structure

### Network Architecture

The model consists of:

* 2 Convolutional Layers
* ReLU Activation Functions
* Max Pooling Layers
* Flatten Layer
* 3 Fully Connected Layers

Architecture Flow:

```text
Input (1 × 32 × 32)

↓ Conv2D (1 → 6, kernel=5)
↓ ReLU
↓ MaxPool

↓ Conv2D (6 → 16, kernel=5)
↓ ReLU
↓ MaxPool

↓ Flatten

↓ Linear (400 → 120)
↓ ReLU

↓ Linear (120 → 84)
↓ ReLU

↓ Linear (84 → 10)

↓ Output
```

---

## Concepts Covered

### 1. Defining Neural Networks

* Creating custom models using `nn.Module`
* Initializing layers in `__init__()`
* Implementing forward propagation in `forward()`

### 2. Forward Pass

* Processing input tensors through layers
* Feature extraction using convolutions
* Producing predictions from learned features

### 3. Automatic Differentiation

* Computational graph creation
* Gradient tracking
* Using `backward()` for gradient computation

### 4. Loss Functions

Implemented:

* Mean Squared Error (`nn.MSELoss`)

Purpose:

* Measure the difference between predictions and targets
* Provide feedback for learning

### 5. Backpropagation

* Gradient computation through the entire network
* Updating parameter gradients automatically

### 6. Optimization

Using:

```python
optimizer = optim.SGD(net.parameters(), lr=0.01)
```

Training steps:

```python
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

---

## Key PyTorch Components

### torch.Tensor

Multi-dimensional array supporting:

* Mathematical operations
* Gradient computation
* GPU acceleration

### nn.Module

Base class for all neural networks.

Provides:

* Parameter management
* Device handling
* Model saving/loading

### nn.Parameter

Special tensor automatically registered as a learnable parameter.

### autograd

PyTorch's automatic differentiation engine that computes gradients during backpropagation.

---

## Training Workflow

1. Define the neural network
2. Load input data
3. Perform a forward pass
4. Compute loss
5. Calculate gradients with backpropagation
6. Update model weights
7. Repeat for multiple epochs

---

## Technologies Used

* Python
* PyTorch
* Torch.nn
* Torch.optim
* Autograd

---

## Skills Gained

* Deep Learning Fundamentals
* CNN Architecture Design
* Forward and Backward Propagation
* Loss Function Implementation
* Gradient-Based Optimization
* PyTorch Model Development



