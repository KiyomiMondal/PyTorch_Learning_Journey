# Day 02 - Understanding Tensors in PyTorch

## Overview

This repository documents my Day 03 progress in learning PyTorch.

Today's focus was on **Tensors**, the fundamental data structure that powers all PyTorch operations. Since tensors are used to store data, model parameters, gradients, and predictions, understanding them is essential before moving into topics like Autograd, Neural Networks, and Deep Learning.

---

## What I Learned

### What are Tensors?

Tensors are multidimensional arrays that are similar to NumPy arrays but come with additional capabilities:

* GPU acceleration support
* Automatic differentiation support
* Efficient mathematical operations
* Deep learning optimization

In PyTorch, tensors are used to represent:

* Input data
* Output predictions
* Model weights
* Gradients during training

---

## Tensor Creation Methods

I explored multiple ways to create tensors:

### From Python Data

```python
data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)
```

### From NumPy Arrays

```python
np_array = np.array(data)
tensor = torch.from_numpy(np_array)
```

### From Existing Tensors

```python
torch.ones_like(tensor)
torch.rand_like(tensor)
```

### Using Shapes

```python
torch.rand(shape)
torch.ones(shape)
torch.zeros(shape)
```

---

## Tensor Attributes

Learned how to inspect tensor properties:

```python
tensor.shape
tensor.dtype
tensor.device
```

Example:

```text
Shape: torch.Size([3, 4])
Data Type: torch.float32
Device: cpu
```

These attributes help us understand how tensors are stored and processed.

---

## Tensor Operations

### Indexing and Slicing

Accessing specific rows, columns, and elements:

```python
tensor[0]
tensor[:, 0]
tensor[..., -1]
```

---

### Tensor Concatenation

Combining multiple tensors using:

```python
torch.cat()
```

Example:

```python
torch.cat([tensor, tensor, tensor], dim=1)
```

---

### Matrix Multiplication

Performed matrix multiplication using:

```python
tensor @ tensor.T
```

and

```python
tensor.matmul(tensor.T)
```

---

### Element-wise Multiplication

```python
tensor * tensor
```

or

```python
tensor.mul(tensor)
```

---

### Aggregation Operations

Converting tensor values into Python numbers:

```python
tensor.sum()
tensor.item()
```

---

## GPU / Accelerator Support

Learned how tensors can be moved to hardware accelerators for faster computation.

```python
tensor.to(device)
```

Supported devices include:

* CUDA (NVIDIA GPUs)
* MPS (Apple Silicon)
* XPU
* MTIA
* CPU

This capability is one of the major reasons PyTorch is widely used in Deep Learning.

---

## In-Place Operations

Explored operations that directly modify tensor values:

```python
tensor.add_(5)
```

Important takeaway:

* In-place operations save memory.
* They can interfere with gradient calculations.
* Use them carefully during model training.

---

## NumPy and PyTorch Interoperability

One of the most interesting concepts I learned today was that PyTorch tensors and NumPy arrays can share memory.

### Tensor → NumPy

```python
numpy_array = tensor.numpy()
```

### NumPy → Tensor

```python
tensor = torch.from_numpy(numpy_array)
```

Changes in one object are reflected in the other when memory is shared.

Example:

```python
tensor.add_(1)
```

The corresponding NumPy array updates automatically.

---

## Key Takeaways

* Tensors are the foundation of PyTorch.
* They are similar to NumPy arrays but support GPU acceleration.
* Tensor operations are optimized for Deep Learning workflows.
* Understanding tensor manipulation is crucial before learning Autograd and Neural Networks.
* PyTorch and NumPy integrate seamlessly through shared memory.

---

## Skills Practiced

✔ Tensor Creation

✔ Tensor Attributes

✔ Indexing and Slicing

✔ Tensor Concatenation

✔ Matrix Operations

✔ Element-wise Operations

✔ Aggregation Functions

✔ In-Place Operations

✔ GPU Support

✔ NumPy ↔ PyTorch Conversion

---


## About This Repository

This repository is part of my public PyTorch learning journey where I document concepts, code implementations, experiments, and key takeaways while building a strong foundation in Deep Learning and AI Engineering.

# 📚 PyTorch Learning Journey - Day 02: Datasets & DataLoaders

## 🎯 Overview

This repository contains my Day 02 learning notes and code while studying PyTorch from the official PyTorch documentation.

In this notebook, I explored how PyTorch handles data using **Dataset** and **DataLoader**, two essential components for preparing and feeding data into machine learning models.

The examples use the **FashionMNIST** dataset and demonstrate how to load datasets, create custom datasets, visualize samples, and prepare mini-batches for model training.

---


# Understanding Datasets

A `Dataset` stores:

* Features (input data)
* Labels (target values)

PyTorch provides many built-in datasets through TorchVision, TorchText, and TorchAudio.

Example datasets include:

* FashionMNIST
* CIFAR-10
* ImageNet
* COCO

---

###  Loading FashionMNIST Dataset

Learned how to:

* Download datasets automatically
* Separate training and testing datasets
* Apply transformations during loading
* Convert images into tensors

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
)
```

---

###  Data Transformations

Used:

```python
v2.ToImage()
v2.ToDtype(torch.float32, scale=True)
```

These transforms:

* Convert images to tensors
* Scale pixel values from 0–255 to 0–1
* Prepare data for neural network training

---

###  Visualizing Dataset Samples

Learned how to:

* Access individual samples
* Retrieve image-label pairs
* Display images using Matplotlib

Example:

```python
img, label = training_data[0]
```

---

###  Creating Custom Datasets

Implemented a custom dataset class by inheriting from:

```python
torch.utils.data.Dataset
```

Required methods:

### `__init__()`

Initializes dataset information.

### `__len__()`

Returns dataset size.

### `__getitem__()`

Loads and returns individual samples.

Example:

```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        ...
```

---

###  Understanding DataLoaders

A `DataLoader` wraps a Dataset and provides:

* Batch loading
* Shuffling
* Efficient iteration
* Multiprocessing support

Example:

```python
train_dataloader = DataLoader(
    training_data,
    batch_size=64,
    shuffle=True
)
```

---

###  Working with Mini-Batches

Learned how to retrieve batches:

```python
train_features, train_labels = next(iter(train_dataloader))
```

Output:

```python
Feature batch shape:
torch.Size([64, 1, 28, 28])

Labels batch shape:
torch.Size([64])
```

This means:

* 64 images per batch
* 1 grayscale channel
* 28 × 28 image size

---

## 🔑 Key Learnings

✅ Difference between Dataset and DataLoader

✅ Loading built-in datasets using TorchVision

✅ Applying transformations to data

✅ Visualizing dataset samples

✅ Building custom datasets

✅ Using mini-batches during training

✅ Understanding shuffling and batching

---

## 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Pandas

---

## 🚀 Why This Matters

Efficient data loading is one of the most important parts of deep learning workflows.

Understanding Datasets and DataLoaders helps in:

* Training models faster
* Managing large datasets
* Creating custom ML pipelines
* Building production-ready deep learning systems

---


### 🌱 Learning Journey

This repository is part of my PyTorch learning series where I document concepts, code implementations, and key takeaways as I progress through Deep Learning with PyTorch.


# Understanding Transforms

## 🎯 Overview

This repository documents my Day 01 learning progress while studying PyTorch from the official PyTorch documentation.

In this notebook, I explored **Transforms**, an essential preprocessing step in Deep Learning pipelines. Transforms help convert raw data into a format suitable for training machine learning and deep learning models.

Using the FashionMNIST dataset, I learned how to transform images into tensors, normalize pixel values, and convert class labels into one-hot encoded vectors.

---

## 📖 Topics Covered

###  What are Transforms?

Real-world data rarely comes in the exact format required for model training.

**Transforms** are used to:

* Preprocess data
* Convert images into tensors
* Normalize values
* Augment datasets
* Transform labels

PyTorch provides built-in transforms through:

```python
torchvision.transforms
```

and the newer:

```python
torchvision.transforms.v2
```

API.

---

###  Feature Transformations

Feature transformations modify the input data (images).

In this project:

```python
transform=v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])
```

#### What happens?

### `v2.ToImage()`

Converts:

* PIL Images
* NumPy Arrays

into:

```python
torchvision.tv_tensors.Image
```

objects.

---

### `v2.ToDtype(torch.float32, scale=True)`

Converts image pixels:

```text
0 - 255
```

to:

```text
0.0 - 1.0
```

and changes datatype to:

```python
torch.float32
```

This normalization helps neural networks train more efficiently.

---

###  Label Transformations

Label transformations modify target values.

FashionMNIST labels are originally integers:

```text
0 → T-Shirt
1 → Trouser
2 → Pullover
...
9 → Ankle Boot
```

For training, labels can be converted into **one-hot encoded vectors**.

Example:

```python
3
```

becomes:

```python
[0,0,0,1,0,0,0,0,0,0]
```

---

###  Lambda Transform

A Lambda Transform allows custom transformation logic.

Example:

```python
target_transform=v2.Lambda(
    lambda y: F.one_hot(
        torch.tensor(y),
        num_classes=10
    ).float()
)
```

This performs:

1. Converts label to tensor
2. Applies one-hot encoding
3. Converts result to float

---

###  Using FashionMNIST Dataset

The dataset is loaded with both feature and label transformations:

```python
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(
            torch.tensor(y),
            num_classes=10
        ).float()
    ),
)
```

---

## 🔑 Key Learnings

✅ Understanding the purpose of data preprocessing

✅ Difference between feature transforms and label transforms

✅ Using `torchvision.transforms.v2`

✅ Converting images into tensors

✅ Normalizing image pixel values

✅ One-hot encoding labels

✅ Applying custom transformations with Lambda

✅ Preparing datasets for neural network training

---

## 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* TorchVision Transforms v2
* FashionMNIST Dataset

---

## 🚀 Why Transforms Matter

Transforms are a critical step in any Deep Learning workflow because they:

* Standardize input data
* Improve model performance
* Reduce training instability
* Prepare labels for loss functions
* Enable data augmentation techniques

Without proper preprocessing, neural networks often struggle to learn effectively.

---

## 🌱 Learning Journey

This repository is part of my PyTorch learning series where I document concepts, code implementations, and practical examples as I build a strong foundation in Deep Learning with PyTorch.
