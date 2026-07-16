# Day 5 - PyTorch Autograd Tutorial

## Overview

This project explores **PyTorch Autograd**, the automatic differentiation engine that powers gradient computation and backpropagation in deep learning models.

The notebook demonstrates how Autograd tracks tensor operations, builds computational graphs dynamically, computes gradients automatically, and enables efficient neural network training.

## Topics Covered

* Introduction to Automatic Differentiation
* Understanding Computational Graphs
* Gradient Tracking with `requires_grad`
* Using `.backward()` for Gradient Computation
* Inspecting Gradients with `.grad`
* Working with `grad_fn`
* Autograd in Neural Network Training
* Gradient Accumulation and `optimizer.zero_grad()`
* Disabling Gradient Tracking with:

  * `torch.no_grad()`
  * `detach()`
  * `requires_grad=False`
* In-place Operations and Autograd
* Autograd Profiler
* Jacobian and Hessian Computation
* Vector-Jacobian Products (VJP)
* Jacobian-Vector Products (JVP)

## Learning Objectives

By completing this notebook, you will learn how to:

* Understand the role of gradients in deep learning.
* Use PyTorch's Autograd engine effectively.
* Build and inspect computational graphs.
* Compute derivatives automatically.
* Train neural networks using backpropagation.
* Manage gradient accumulation correctly.
* Optimize memory and computation during inference.

## Key Concepts

### Gradient Tracking

Autograd records operations performed on tensors that have:

```python
requires_grad=True
```

This enables PyTorch to automatically compute gradients during the backward pass.

### Backpropagation

Gradients are computed using:

```python
loss.backward()
```

The calculated gradients are stored in the tensor's:

```python
tensor.grad
```

attribute.

### Gradient Reset

Before every training iteration, gradients should be cleared:

```python
optimizer.zero_grad()
```

to prevent unwanted accumulation.

## Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 2

y.backward()

print(x.grad)
```

Output:

```python
tensor(7.)
```

Since:

```
y = x² + 3x + 2
dy/dx = 2x + 3
```

At `x = 2`:

```
dy/dx = 7
```

## Requirements

* Python 3.10+
* PyTorch
* Matplotlib

Install dependencies:

```bash
pip install torch matplotlib
```

# PyTorch Modules, Layers & Functions

## Overview

This notebook explores the fundamental building blocks of neural networks in PyTorch. It covers the concepts of `torch.nn.Module`, `torch.nn.Parameter`, common neural network layers, activation functions, loss functions, and other important components used in deep learning.

The goal of this notebook is to provide a beginner-friendly understanding of how PyTorch models are structured and how different layers work together to build powerful neural networks.

---

## Topics Covered

### 1. `torch.nn.Module` and `torch.nn.Parameter`

* Understanding the role of `nn.Module`
* Learnable parameters in PyTorch
* Parameter registration
* Accessing model parameters using `parameters()`
* Building a simple custom model

### 2. Common Layer Types

#### Linear (Fully Connected) Layers

* Input and output features
* Weight matrices and biases
* Forward propagation through linear layers

#### Convolutional Layers (CNNs)

* Kernels and feature maps
* Input and output channels
* Convolution operations
* LeNet architecture example

#### Recurrent Layers (RNN, LSTM, GRU)

* Sequential data processing
* Hidden states and memory
* LSTM-based sequence modeling

#### Transformer Layers

* Attention mechanisms
* Encoder-decoder architecture
* Modern NLP applications

### 3. Other Important Layers

#### Max Pooling

* Spatial downsampling
* Feature extraction

#### Batch Normalization

* Data normalization
* Faster and more stable training

#### Dropout

* Regularization technique
* Preventing overfitting

### 4. Activation Functions

* ReLU
* Leaky ReLU
* Tanh
* Sigmoid
* GELU
* ELU
* Softmax

### 5. Loss Functions

* Mean Squared Error (MSELoss)
* Cross Entropy Loss
* Negative Log Likelihood Loss (NLLLoss)
* Binary Cross Entropy (BCELoss)
* L1 Loss

---

## Learning Objectives

After completing this notebook, you will be able to:

* Understand how PyTorch models are organized.
* Create custom neural network architectures.
* Differentiate between various layer types.
* Understand the purpose of activation functions.
* Choose appropriate loss functions for different tasks.
* Apply pooling, normalization, and dropout layers effectively.
* Build a strong foundation for advanced deep learning topics.

---

## Example Model Structure

```python
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(100, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
```

---

## Key Takeaways

* `nn.Module` is the foundation of all PyTorch models.
* `nn.Parameter` represents learnable weights and biases.
* Linear layers are commonly used for classification tasks.
* Convolutional layers excel at image processing.
* Recurrent layers handle sequential data.
* Transformers power modern NLP and generative AI systems.
* Pooling, normalization, and dropout improve model performance.
* Activation functions introduce non-linearity.
* Loss functions guide the learning process.

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision matplotlib
```

