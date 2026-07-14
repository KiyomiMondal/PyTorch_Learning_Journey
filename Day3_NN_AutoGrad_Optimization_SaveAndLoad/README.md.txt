# Day 3 of Learning PyTorch – Building a Neural Network

## Overview
This project is part of my PyTorch learning journey. On Day 3, I explored how neural networks are built in PyTorch using the `torch.nn` module and implemented a simple feedforward neural network for image classification on the FashionMNIST dataset.

The focus of this tutorial was understanding the architecture of neural networks, model layers, activation functions, model parameters, and how data flows through a network during prediction.

---

## What I Learned

### Device Selection
- Learned how to detect and use available hardware accelerators.
- Used GPU (CUDA) when available for faster computation.
- Understood the importance of moving models and tensors to the correct device.

### Building a Neural Network
- Created a custom neural network by subclassing `nn.Module`.
- Implemented the model architecture inside the `__init__()` method.
- Defined the forward pass using the `forward()` method.

### Understanding Core Layers

#### Flatten Layer
- Used `nn.Flatten()` to convert 28×28 images into a 784-dimensional vector.
- Learned why neural networks require flattened input before fully connected layers.

#### Linear Layers
- Used `nn.Linear()` layers to perform linear transformations.
- Explored how weights and biases are applied to inputs.

#### ReLU Activation
- Learned how `nn.ReLU()` introduces non-linearity.
- Observed how negative values are converted to zero.

#### Sequential Container
- Used `nn.Sequential()` to stack layers in a simple and organized manner.
- Understood how data automatically flows through each layer.

#### Softmax Layer
- Converted raw output logits into probability distributions.
- Used probabilities to determine the predicted class.

### Model Predictions
- Generated random sample inputs.
- Passed inputs through the neural network.
- Converted logits to probabilities using Softmax.
- Obtained final predictions using `argmax()`.

### Exploring Model Parameters
- Inspected model weights and biases using:
  - `model.parameters()`
  - `model.named_parameters()`
- Learned how PyTorch automatically tracks trainable parameters.

---

## Neural Network Architecture

Input Image (28×28)

↓ Flatten

784 Features

↓ Linear Layer (784 → 512)

↓ ReLU

↓ Linear Layer (512 → 512)

↓ ReLU

↓ Linear Layer (512 → 10)

↓ Output Logits

---

## Key Concepts Covered
- Neural Networks
- `torch.nn.Module`
- Forward Propagation
- Device Management (CPU/GPU)
- Flatten Layer
- Fully Connected Layers
- ReLU Activation Function
- Softmax Probabilities
- Sequential Models
- Model Parameters and Weights

---

## Technologies Used
- Python
- PyTorch
- TorchVision
- CUDA (GPU Acceleration)

---

## Learning Outcome
By completing this tutorial, I gained a solid understanding of how neural networks are structured in PyTorch, how layers interact with one another, and how predictions are generated from input data. This forms the foundation for training, optimization, and more advanced deep learning architectures in future projects.


---

# Automatic Differentiation with `torch.autograd` 

This repository contains my **Day 3 of Learning PyTorch** notes and practice on **Automatic Differentiation (Autograd)**, one of the core concepts behind training neural networks.

## 📖 Topics Covered

### 1. Introduction to Autograd

* Understanding backpropagation in neural networks
* Role of gradients in updating model parameters
* How PyTorch automatically computes gradients using `torch.autograd`

### 2. Computational Graph

* Building a computational graph using tensors and operations
* Understanding parameters (`w`, `b`) and `requires_grad=True`
* Exploring `grad_fn` and how PyTorch tracks operations

### 3. Computing Gradients

* Using `loss.backward()`
* Accessing gradients through `.grad`
* Understanding gradient calculation for model parameters

### 4. Gradient Tracking Control

* Using `torch.no_grad()`
* Using `.detach()`
* When and why gradient tracking should be disabled

### 5. Dynamic Computational Graphs

* Understanding Directed Acyclic Graphs (DAGs)
* How PyTorch recreates the graph after every backward pass
* Benefits of dynamic graphs for flexible model architectures

### 6. Gradient Accumulation

* Why gradients accumulate by default
* Importance of clearing gradients before the next backward pass
* Introduction to gradient management during training

### 7. Jacobian Products

* Understanding vector-valued functions
* Introduction to Jacobian matrices
* Computing Jacobian-vector products using `backward()`

---

## 🛠 Technologies Used

* Python
* PyTorch (`torch.autograd`)

---

## 🎯 Key Learnings

* Autograd automatically computes gradients required for training neural networks.
* Computational graphs help PyTorch track all operations on tensors.
* `loss.backward()` performs backpropagation and computes gradients.
* Gradient tracking can be disabled using `torch.no_grad()` or `detach()` for faster inference.
* Gradients accumulate across backward passes and should be reset when necessary.
* PyTorch uses dynamic computational graphs, making model development highly flexible.

---

## 📚 Learning Journey

This repository is part of my **PyTorch Learning Series**, where I document concepts, code examples, handwritten notes, and key takeaways as I progress through deep learning fundamentals.

**Day 3 Focus:** Automatic Differentiation with `torch.autograd` 🚀

---



# Optimizing Model Parameters in PyTorch

## 📌 Project Overview

This project explores one of the most important stages of deep learning: **optimizing model parameters**. After building a neural network and preparing the dataset, the next step is training the model so that it learns patterns from the data.

Using the **FashionMNIST** dataset, this project demonstrates how PyTorch performs model training through **forward propagation, loss calculation, backpropagation, and parameter optimization**. It introduces key concepts such as hyperparameters, loss functions, optimizers, training loops, and evaluation loops.

---

## 🎯 Learning Objectives

Through this project, I learned:

* How neural networks learn from data
* The role of loss functions in measuring prediction error
* How gradient descent updates model parameters
* The importance of optimizers in training
* How training and testing loops work
* How hyperparameters affect model performance
* The complete workflow of model optimization in PyTorch

---

## 🧠 Key Concepts Covered

### 1. Neural Network Architecture

A simple feedforward neural network was created using:

* `nn.Flatten()`
* `nn.Linear()`
* `nn.ReLU()`

Architecture:

```text
Input Image (28×28)
       ↓
Flatten Layer
       ↓
Linear (784 → 512)
       ↓
ReLU
       ↓
Linear (512 → 512)
       ↓
ReLU
       ↓
Linear (512 → 10)
       ↓
Output Logits
```

---

### 2. Hyperparameters

The model training process was controlled using:

| Hyperparameter | Value | Purpose                           |
| -------------- | ----- | --------------------------------- |
| Learning Rate  | 0.001 | Controls update size              |
| Batch Size     | 64    | Samples processed per batch       |
| Epochs         | 10    | Number of complete dataset passes |

---

### 3. Loss Function

The project uses:

```python
nn.CrossEntropyLoss()
```

Purpose:

* Measures prediction error
* Compares model output with actual labels
* Guides the learning process

Lower loss indicates better predictions.

---

### 4. Optimizer

The optimizer used is:

```python
torch.optim.SGD()
```

SGD (Stochastic Gradient Descent) updates model weights based on calculated gradients.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)
```

---

### 5. Backpropagation

PyTorch automatically computes gradients using Autograd.

Training steps:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

#### Workflow

1. Forward Pass
2. Calculate Loss
3. Backward Pass
4. Update Parameters
5. Reset Gradients

---

### 6. Training Loop

The training loop:

* Loads batches from DataLoader
* Generates predictions
* Computes loss
* Performs backpropagation
* Updates model parameters

```python
model.train()
```

Training mode enables proper behavior of layers such as Dropout and BatchNorm.

---

### 7. Testing Loop

The test loop evaluates model performance without updating weights.

```python
model.eval()
```

and

```python
with torch.no_grad():
```

Benefits:

* Faster evaluation
* Reduced memory usage
* No gradient computation

---

## 🔄 Optimization Workflow

```text
Input Data
      ↓
Forward Pass
      ↓
Prediction
      ↓
Loss Function
      ↓
Loss Value
      ↓
Backward Pass
      ↓
Gradients
      ↓
Optimizer
      ↓
Updated Weights
      ↓
Better Predictions
```

---

## 📊 Results

During training:

* Loss gradually decreased
* Accuracy steadily improved
* Model learned meaningful patterns from FashionMNIST

Example output:

```text
Epoch 1
Accuracy: 46.3%
Avg Loss: 2.16

Epoch 2
Accuracy Improved
Loss Reduced
```

The model continues improving as more epochs are completed.

---

## 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* FashionMNIST Dataset
* DataLoader
* Neural Networks
* SGD Optimizer
* CrossEntropyLoss

---

## 📚 Key Takeaways

* Training is an iterative optimization process.
* Loss functions measure prediction errors.
* Optimizers update weights to reduce loss.
* Backpropagation calculates gradients automatically.
* Hyperparameters significantly impact training performance.
* Training and testing loops are the core of every deep learning workflow.

---

## 🚀 What I Learned

This project helped me understand the complete training pipeline of a neural network in PyTorch, including:

* Model optimization
* Gradient descent
* Loss calculation
* Parameter updates
* Training vs Evaluation modes
* Building reusable training and testing loops

It provided a strong foundation for training more advanced deep learning models in future projects.

---

# Save and Load Models in PyTorch

## 📌 Overview

This project demonstrates how to save and load deep learning models in PyTorch. It covers two common approaches:

1. **Saving and Loading Model Weights (`state_dict`)** – Recommended Best Practice
2. **Saving and Loading the Entire Model** – Includes both architecture and weights

Understanding model persistence is essential because trained models can be reused later without retraining, saving both time and computational resources.

---

## 🎯 Learning Objectives

By completing this tutorial, you will learn:

* What a `state_dict` is in PyTorch
* How to save trained model weights
* How to load saved weights into a model
* Why `model.eval()` is important during inference
* How to save and load an entire model architecture
* Best practices for model serialization

---

## 📚 Concepts Covered

### 1. Model State Dictionary (`state_dict`)

A PyTorch model stores all learnable parameters (weights and biases) inside a dictionary called **state_dict**.

Example:

```python
model.state_dict()
```

The state dictionary contains:

* Layer weights
* Layer biases
* Batch normalization parameters
* Other learnable parameters

---

### 2. Saving Model Weights

The recommended approach is to save only the model parameters.

```python
import torchvision.models as models
import torch

model = models.vgg16(weights='IMAGENET1K_V1')

torch.save(model.state_dict(), "model_weights.pth")
```

This creates a file:

```text
model_weights.pth
```

containing the trained weights.

---

### 3. Loading Model Weights

To load saved weights:

```python
model = models.vgg16()

model.load_state_dict(
    torch.load("model_weights.pth", weights_only=True)
)

model.eval()
```

### Why create the model first?

The weights file contains only parameter values, not the architecture. Therefore, PyTorch must know the model structure before loading the weights.

---

### 4. Evaluation Mode

Before making predictions:

```python
model.eval()
```

This:

* Disables Dropout
* Uses BatchNorm running statistics
* Produces consistent predictions

Without it, inference results may vary.

---

### 5. Saving the Entire Model

Instead of saving only weights, you can save the complete model.

```python
torch.save(model, "model.pth")
```

This stores:

* Model architecture
* Model weights
* Configuration information

---

### 6. Loading the Entire Model

```python
model = torch.load(
    "model.pth",
    weights_only=False
)
```

After loading, the model is ready to use.

---

## ⚠️ Important Notes

### Recommended Practice

✅ Save only the state dictionary:

```python
torch.save(model.state_dict(), "model_weights.pth")
```

because it is:

* More flexible
* More portable
* Less prone to compatibility issues

---

### When Saving the Entire Model

PyTorch uses Python's **pickle** module.

Because of this:

* The original model class must be available when loading.
* Changes in code structure may break loading.
* Less portable across projects.

---

## 🔄 Workflow Summary

```text
Train Model
     │
     ▼
Save state_dict()
     │
     ▼
Create Same Model Architecture
     │
     ▼
Load state_dict()
     │
     ▼
model.eval()
     │
     ▼
Inference / Prediction
```

---

## 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* VGG16 Pretrained Model

---

## 📖 Key Takeaways

* `state_dict()` stores model parameters.
* `torch.save()` saves model weights or entire models.
* `torch.load()` restores saved information.
* Always use `model.eval()` before inference.
* Saving only weights is considered the industry-standard practice.
* Saving the full model is possible but less flexible.

---

### 🚀 Learning PyTorch 

This notebook is part of my PyTorch learning journey where I explored model serialization, model persistence, and best practices for saving and loading neural networks efficiently.
