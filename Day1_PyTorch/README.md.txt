# Day 01 - Building My First Neural Network with PyTorch

## Overview

Today I started my PyTorch learning journey by building and training a neural network on the FashionMNIST dataset.

The objective was to understand the complete deep learning workflow in PyTorch, including:

* Loading and preprocessing data
* Creating DataLoaders
* Building a neural network
* Training the model
* Evaluating performance
* Saving and loading trained models
* Making predictions on unseen data

---

## Dataset

### FashionMNIST

FashionMNIST is a collection of grayscale clothing images.

Dataset Details:

* Training Images: 60,000
* Test Images: 10,000
* Image Size: 28 × 28 pixels
* Classes: 10

Classes:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

## Key Concepts Learned

### Dataset

`Dataset` stores the samples and their corresponding labels.

Example:

```python
training_data = datasets.FashionMNIST(...)
```

---

### DataLoader

`DataLoader` creates batches of data and makes dataset iteration easier.

Example:

```python
train_dataloader = DataLoader(training_data, batch_size=64)
```

Batch Output:

```python
Shape of X: [64, 1, 28, 28]
Shape of y: [64]
```

Understanding the dimensions:

* 64 = Batch Size
* 1 = Number of Channels (Grayscale)
* 28 = Height
* 28 = Width

---

## Neural Network Architecture

The model consists of:

```text
Input Image (28×28)

↓

Flatten Layer

↓

784 Features

↓

Linear Layer (784 → 512)

↓

ReLU

↓

Linear Layer (512 → 512)

↓

ReLU

↓

Linear Layer (512 → 10)

↓

Class Scores
```

PyTorch Model:

```python
nn.Linear(28*28, 512)
nn.ReLU()

nn.Linear(512, 512)
nn.ReLU()

nn.Linear(512, 10)
```

---

## Concepts Understood

### Flatten Layer

Converts:

```text
28 × 28 Image
```

into

```text
784 Features
```

because:

```text
28 × 28 = 784
```

---

### Linear Layer

Performs:

```text
Output = Weight × Input + Bias
```

The model learns these weights during training.

---

### ReLU Activation

Formula:

```text
f(x) = max(0, x)
```

Purpose:

* Removes negative values
* Helps neural networks learn complex patterns

Example:

```text
Input:
[-5, 3, -2, 8]

Output:
[0, 3, 0, 8]
```

---

## Training Process

### Loss Function

```python
nn.CrossEntropyLoss()
```

Used for multi-class classification problems.

Measures how wrong the predictions are.

---

### Optimizer

```python
torch.optim.SGD()
```

SGD = Stochastic Gradient Descent

Purpose:

* Updates weights
* Reduces prediction error

---

### Training Workflow

```text
Input Data

↓

Forward Pass

↓

Prediction

↓

Calculate Loss

↓

Backpropagation

↓

Update Weights

↓

Repeat
```

---

## Model Performance

Training was conducted for:

```text
5 Epochs
```

Results:

| Epoch | Accuracy |
| ----- | -------- |
| 1     | 36.5%    |
| 2     | 53.1%    |
| 3     | 61.1%    |
| 4     | 63.7%    |
| 5     | 65.2%    |

Observation:

* Accuracy increased every epoch.
* Loss continuously decreased.
* The model successfully learned patterns from the dataset.

---

## Saving the Model

Saved model weights using:

```python
torch.save(model.state_dict(), "model.pth")
```

Benefits:

* Reuse trained models
* Avoid retraining from scratch

---

## Loading the Model

Loaded saved weights using:

```python
model.load_state_dict(torch.load("model.pth"))
```

Successfully restored the trained model.

---

## Prediction Example

Model Prediction:

```text
Predicted: Ankle boot
Actual: Ankle boot
```

The model correctly classified the test image.

---

## Challenges Faced

* Understanding tensor shapes
* Understanding DataLoader batches
* Understanding how forward propagation works
* Understanding how loss and optimization interact

---

## Key Takeaways

* Dataset stores data while DataLoader creates batches.
* Neural networks are built using layers and activation functions.
* Training is an iterative process of reducing prediction errors.
* Backpropagation updates model weights automatically.
* Loss should decrease while accuracy should increase.
* PyTorch provides a clean and intuitive workflow for deep learning.

---

## Files Included

```text
Day-01-FashionMNIST-Basics/
│
├── notebook.ipynb
├── README.md
├── handwritten_notes.png
└── model.pth
```

---

## Day 01 Summary

✅ Learned Dataset and DataLoader

✅ Built a Neural Network using nn.Module

✅ Trained a model on FashionMNIST

✅ Learned Loss Functions and Optimizers

✅ Evaluated model performance

✅ Saved and Loaded model weights

✅ Made predictions using the trained model

---

### Next Learning Goals

* PyTorch Tensors
* Tensor Operations
* Autograd
* Gradient Tracking
* GPU Acceleration
* Convolutional Neural Networks (CNNs)
