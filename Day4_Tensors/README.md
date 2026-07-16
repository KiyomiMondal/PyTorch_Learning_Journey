# Day 04 - PyTorch Fundamentals

This repository contains my hands-on implementations and notes while learning PyTorch from the official PyTorch tutorials.

## Overview

In this project, I explored the fundamental building blocks of PyTorch, including:

* Creating and manipulating tensors
* Performing mathematical operations on tensors
* Understanding random initialization and reproducibility
* Building neural network architectures using `torch.nn.Module`
* Implementing the LeNet convolutional neural network
* Working with datasets and dataloaders
* Loading and preprocessing the CIFAR-10 dataset
* Normalizing image data using dataset statistics
* Understanding CPU and GPU tensor operations

## Topics Covered

### 1. PyTorch Tensors

* Tensor creation (`zeros`, `ones`, `rand`)
* Tensor datatypes
* Arithmetic operations
* Matrix operations
* Statistical functions
* Random seed initialization

### 2. Neural Networks in PyTorch

* Creating custom models with `nn.Module`
* Defining layers in `__init__()`
* Implementing forward propagation
* Understanding convolutional and fully connected layers
* LeNet architecture implementation

### 3. Datasets and DataLoaders

* Loading CIFAR-10 using TorchVision
* Image transformations
* Tensor conversion
* Data normalization
* Batch processing

### 4. Data Preprocessing

Computed CIFAR-10 dataset statistics:

```python
Mean: [0.4914, 0.4822, 0.4465]
Std : [0.2470, 0.2435, 0.2616]
```

These values were used to normalize the dataset before training.

## Technologies Used

* Python
* PyTorch
* TorchVision
* Jupyter Notebook

## Learning Outcome

Through this project, I gained practical experience with PyTorch fundamentals and developed a strong understanding of tensors, neural network construction, data preprocessing, and dataset handling, which form the foundation for building deep learning models.


# PyTorch Tensors Fundamentals

This repository contains my hands-on implementation and notes while learning the fundamentals of PyTorch tensors from the official PyTorch tutorials.

## Overview

Tensors are the core data structure in PyTorch and form the foundation of all deep learning operations. In this notebook, I explored tensor creation, manipulation, mathematical operations, broadcasting, reshaping, device management, and interoperability with NumPy.

## Topics Covered

### Tensor Creation

* `torch.empty()`
* `torch.zeros()`
* `torch.ones()`
* `torch.rand()`
* `torch.tensor()`
* `*_like()` tensor creation methods

### Randomness and Reproducibility

* Setting random seeds using `torch.manual_seed()`
* Generating reproducible random tensors

### Tensor Properties

* Tensor shapes
* Dimensions and ranks
* Data types (`dtype`)
* Device information

### Tensor Operations

* Arithmetic operations
* Mathematical functions
* Trigonometric functions
* Reduction operations
* Linear algebra operations
* Matrix multiplication

### Broadcasting

* Broadcasting rules
* Element-wise operations on tensors with compatible shapes
* Practical broadcasting examples

### In-Place Operations

* Using methods with `_` suffix
* Memory-efficient tensor updates

### Tensor Copying

* Tensor assignment vs cloning
* `clone()`
* `detach()`
* Gradient tracking considerations

### Device Management

* CPU tensors
* GPU/CUDA tensors
* Moving tensors between devices
* Device-aware programming practices

### Tensor Shape Manipulation

* `squeeze()`
* `unsqueeze()`
* `reshape()`
* Batch dimension handling

### NumPy Integration

* Converting NumPy arrays to tensors
* Converting tensors to NumPy arrays
* Shared memory behavior

## Technologies Used

* Python
* PyTorch
* NumPy
* Jupyter Notebook

## Key Learnings

* Understanding tensors as multidimensional arrays.
* Creating and manipulating tensors efficiently.
* Performing mathematical and linear algebra operations.
* Using broadcasting for flexible tensor computations.
* Managing tensor memory and in-place operations.
* Working with CPU and GPU devices.
* Reshaping tensors for deep learning workflows.
* Interoperating seamlessly with NumPy.


