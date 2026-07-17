# Day 6 - PyTorch Fundamentals: Neural Networks & Autograd

## Overview

This repository contains hands-on implementations and notes covering some of the most important concepts in PyTorch, including neural network construction, common layer types, automatic differentiation (Autograd), and the training workflow used in deep learning.

The project serves as a practical learning resource for understanding how PyTorch models are built, trained, and optimized.

---

## Topics Covered

### 1. `torch.nn.Module` and `torch.nn.Parameter`

* Building custom neural network architectures
* Understanding model structure
* Learnable parameters and weight registration
* Forward propagation

### 2. Linear (Fully Connected) Layers

* Creating dense neural network layers
* Weight and bias parameters
* Input-output transformations

### 3. Convolutional Neural Networks (CNNs)

* Convolutional layers (`Conv2d`)
* Feature extraction from images
* Max pooling operations
* LeNet architecture implementation

### 4. Recurrent Neural Networks (RNNs)

* Sequential data processing
* LSTM networks
* Word embeddings
* Sequence tagging examples

### 5. Transformer Networks

* Transformer architecture overview
* Encoder-decoder structure
* Attention-based learning

### 6. Data Manipulation Layers

* Max Pooling
* Batch Normalization
* Dropout Regularization

### 7. Activation Functions

* ReLU
* Sigmoid
* Tanh
* Softmax

### 8. Loss Functions

* Mean Squared Error (MSE)
* Cross Entropy Loss
* Negative Log Likelihood Loss

### 9. PyTorch Autograd

* Automatic differentiation
* Computational graphs
* Gradient computation
* Backpropagation

### 10. Training Neural Networks

* Forward pass
* Loss calculation
* Backward propagation
* Optimizer updates
* Gradient management

### 11. Advanced Autograd Features

* Jacobian computation
* Hessian computation
* Vector-Jacobian products
* High-level functional API

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib

---

## Project Structure

```text
├── notebooks/
│   └── pytorch_fundamentals.ipynb
├── images/
├── README.md
└── requirements.txt
```

---

## Key Learning Outcomes

After completing this project, you will understand:

* How PyTorch models are structured using `nn.Module`
* How neural network layers interact during forward propagation
* The role of activation functions and loss functions
* How automatic differentiation powers backpropagation
* How gradients are computed and applied during training
* Best practices for managing model parameters and optimizers

---

## Example Workflow

```python
prediction = model(input_data)

loss = criterion(prediction, target)

loss.backward()

optimizer.step()

optimizer.zero_grad()
```

---

## Why This Project?

Understanding `torch.nn` and `torch.autograd` is essential for building deep learning models in PyTorch. These concepts form the foundation of:

* Computer Vision
* Natural Language Processing
* Generative AI
* Large Language Models (LLMs)
* Reinforcement Learning

This repository documents the learning journey through these fundamental building blocks of modern AI systems.


# PyTorch Training with Dataset, DataLoader, Loss Functions, and Optimizers

## Overview

This project demonstrates a complete deep learning workflow in PyTorch using the Fashion-MNIST dataset. It covers essential concepts required for training neural networks, including data loading, preprocessing, model architecture design, loss calculation, optimization, validation, model checkpointing, and experiment tracking with TensorBoard.

The project uses a convolutional neural network (CNN) inspired by the classic LeNet-5 architecture to classify clothing items from the Fashion-MNIST dataset.

---

## Learning Objectives

Through this project, you will learn:

* How to work with `Dataset` and `DataLoader`
* Image preprocessing using `torchvision.transforms.v2`
* Building CNNs with `torch.nn.Module`
* Using loss functions for classification tasks
* Applying optimization algorithms for model training
* Implementing a complete training and validation loop
* Monitoring experiments with TensorBoard
* Saving and loading trained models

---

## Dataset

### Fashion-MNIST

Fashion-MNIST is a collection of grayscale images representing 10 categories of clothing items.

Classes include:

1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

Dataset Statistics:

| Split      | Samples |
| ---------- | ------- |
| Training   | 60,000  |
| Validation | 10,000  |

---

## Data Preprocessing

The following transformations are applied:

```python
v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])
```

### Purpose

* Convert images to tensors
* Scale pixel values to the range [0, 1]
* Normalize images for stable training

---

## Data Loading

PyTorch's `DataLoader` is used to create mini-batches for efficient training.

```python
training_loader = DataLoader(
    training_set,
    batch_size=4,
    shuffle=True
)
```

### Benefits

* Automatic batching
* Data shuffling
* Efficient memory usage
* Easy integration with training loops

---

## Model Architecture

The model is a convolutional neural network inspired by LeNet-5.

### Layers

| Layer   | Description                    |
| ------- | ------------------------------ |
| Conv1   | 1 → 6 channels, kernel size 5  |
| MaxPool | 2×2 pooling                    |
| Conv2   | 6 → 16 channels, kernel size 5 |
| MaxPool | 2×2 pooling                    |
| FC1     | 256 → 120                      |
| FC2     | 120 → 84                       |
| FC3     | 84 → 10                        |

### Activation Function

```python
F.relu()
```

### Output

The network predicts probabilities for 10 clothing categories.

---

## Loss Function

### Cross-Entropy Loss

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

Cross-Entropy Loss is commonly used for multi-class classification problems.

It compares:

* Model predictions
* Ground truth labels

Lower loss values indicate better predictions.

---

## Optimizer

### Stochastic Gradient Descent (SGD)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)
```

### Hyperparameters

* Learning Rate: `0.001`
* Momentum: `0.9`

### Purpose

The optimizer updates model weights using gradients computed during backpropagation.

---

## Training Process

For each batch:

1. Load input images and labels
2. Reset gradients
3. Perform forward propagation
4. Calculate loss
5. Compute gradients using backpropagation
6. Update model parameters
7. Record training metrics

The training function reports loss every 1000 batches.

---

## Validation

After each epoch:

* Model switches to evaluation mode
* Validation data is processed
* Validation loss is calculated
* Results are logged to TensorBoard

```python
model.eval()
with torch.no_grad():
```

This prevents gradient computation and reduces memory consumption.

---

## TensorBoard Integration

Training and validation metrics are logged using TensorBoard.

```python
writer = SummaryWriter(...)
```

TensorBoard helps visualize:

* Training loss
* Validation loss
* Learning progress
* Model performance trends

---

## Model Checkpointing

The model with the lowest validation loss is automatically saved.

```python
torch.save(
    model.state_dict(),
    model_path
)
```

This ensures the best-performing version is preserved.

---

## Training Results

Over 5 epochs, both training and validation losses consistently decrease, demonstrating successful learning.

### Final Results

| Metric                | Value  |
| --------------------- | ------ |
| Final Training Loss   | ~0.296 |
| Final Validation Loss | ~0.342 |

The model achieves strong classification performance on the Fashion-MNIST dataset.

---

## Loading a Saved Model

```python
saved_model = GarmentClassifier()
saved_model.load_state_dict(torch.load(PATH))
```

Once loaded, the model can be used for:

* Further training
* Inference
* Performance evaluation
* Model analysis

---

## Key Concepts Covered

* PyTorch Dataset
* PyTorch DataLoader
* CNN Architecture
* Fashion-MNIST Classification
* Cross-Entropy Loss
* Stochastic Gradient Descent
* Backpropagation
* Validation Loops
* TensorBoard Logging
* Model Checkpointing

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* TensorBoard
* NumPy
* Matplotlib

---

## Conclusion

This project provides a complete introduction to training convolutional neural networks in PyTorch. It demonstrates the end-to-end workflow of loading data, preprocessing images, building a neural network, optimizing model parameters, validating performance, visualizing metrics, and saving trained models. These concepts form the foundation for developing more advanced deep learning applications.

# PyTorch Learning Journey – Training Neural Networks with FashionMNIST

## Overview

This notebook is part of my PyTorch learning journey, where I explored the complete workflow of training a neural network using PyTorch. The project demonstrates how datasets are loaded, how models are trained using loss functions and optimizers, and how training progress can be monitored using TensorBoard.

The implementation uses the FashionMNIST dataset and a Convolutional Neural Network (CNN) inspired by the LeNet-5 architecture for image classification.

---

## Learning Objectives

Through this notebook, I learned:

* How to work with `Dataset` and `DataLoader`
* Image preprocessing using `torchvision.transforms.v2`
* Building custom neural networks with `torch.nn.Module`
* Understanding and applying loss functions
* Using optimizers for gradient-based learning
* Implementing a complete training loop
* Performing model validation
* Tracking training metrics with TensorBoard
* Saving and loading trained models

---

## Dataset

### FashionMNIST

FashionMNIST is a collection of grayscale clothing images belonging to 10 categories:

* T-shirt/Top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle Boot

Dataset Statistics:

| Split      | Samples |
| ---------- | ------- |
| Training   | 60,000  |
| Validation | 10,000  |

---

## Data Preprocessing

The following transformations were applied:

* Convert images to PyTorch tensors
* Scale pixel values to the range [0, 1]
* Normalize images using mean = 0.5 and std = 0.5

```python
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])
```

---

## Model Architecture

A CNN inspired by LeNet-5 was implemented.

### Layers

1. Convolution Layer (1 → 6)
2. Max Pooling
3. Convolution Layer (6 → 16)
4. Max Pooling
5. Fully Connected Layer (256 → 120)
6. Fully Connected Layer (120 → 84)
7. Output Layer (84 → 10)

### Activation Function

* ReLU

### Pooling

* MaxPool2D

---

## Loss Function

Cross Entropy Loss was used for multi-class classification.

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

This loss function compares the predicted class probabilities with the true labels and calculates the prediction error.

---

## Optimizer

Stochastic Gradient Descent (SGD) with Momentum:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)
```

### Hyperparameters

* Learning Rate: 0.001
* Momentum: 0.9

---

## Training Process

For each batch:

1. Load input images and labels
2. Reset gradients
3. Perform forward pass
4. Compute loss
5. Backpropagate gradients
6. Update model parameters

The model was trained for:

```python
EPOCHS = 5
```

---

## Validation

After every epoch:

* Model switched to evaluation mode
* Validation loss computed
* Results logged to TensorBoard
* Best model saved automatically

---

## TensorBoard Integration

Training and validation losses were tracked using TensorBoard.

```python
writer.add_scalars(
    'Training vs. Validation Loss',
    {
        'Training': avg_loss,
        'Validation': avg_vloss
    },
    epoch_number + 1
)
```

TensorBoard helps visualize:

* Training Loss
* Validation Loss
* Learning Progress
* Model Convergence

---

## Results

Training loss consistently decreased over epochs:

| Epoch | Training Loss | Validation Loss |
| ----- | ------------- | --------------- |
| 1     | 0.4298        | 0.4356          |
| 2     | 0.3692        | 0.3711          |
| 3     | 0.3361        | 0.3535          |
| 4     | 0.3099        | 0.3212          |
| 5     | 0.2957        | 0.3418          |

### Key Observation

The model successfully learned meaningful features from FashionMNIST and achieved steadily improving performance throughout training.

---

## Model Saving

The best-performing model was saved automatically:

```python
torch.save(model.state_dict(), model_path)
```

To load a saved model:

```python
saved_model = GarmentClassifier()
saved_model.load_state_dict(torch.load(PATH))
```

---

## Key Concepts Learned

* Dataset vs DataLoader
* Batch Processing
* Data Normalization
* CNN Architecture
* Forward Pass
* Backpropagation
* Cross Entropy Loss
* SGD Optimization
* Validation Workflow
* TensorBoard Logging
* Model Checkpointing

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* TensorBoard
* NumPy
* Matplotlib



