# Autograd Library

This library provides a simple implementation of automatic differentiation and a neural network framework. It is designed to help understand the basics of building neural networks and performing backpropagation using custom tensor operations. It's inspired from [Pytorch](https://github.com/pytorch/pytorch) and Andrej Karpathy's [video](https://youtu.be/VMj-3S1tku0?si=WSwn1KHF6j9qM9Lg).

>**Note**: The tensor class is simple a wrapper and not an actual tensor.

## Overview

The library is divided into two main components:

1. **Core Module (`autograd/core.py`)**: Implements the `tensor` class, which supports basic arithmetic operations, activation functions, automatic differentiation and visualization.
2. **Neural Network Module (`autograd/nn.py`)**: Implements a simple neural network framework with layers and neurons, supporting forward, backward passes and visualization.

## Core Module

### Class `tensor` 

- **Purpose**: Represents a scalar value with automatic differentiation and visualization capabilities.
- **Attributes**:
  - `data`: The scalar value.
  - `grad`: The gradient of the tensor, used for backpropagation.
  - `_parent`: A set of parent tensors involved in the operation that produced this tensor.
  - `_op`: The operation that produced this tensor.
- **Methods**:
  - Arithmetic operations (`__add__`, `__sub__`, `__mul__`, etc.) are overloaded to support tensor operations and track computation graphs.
  - Activation functions (`sigmoid`, `tanh`, `relu`) are implemented with backward methods to compute gradients.
  - `backward()`: Computes the gradient of the tensor with respect to its inputs using reverse-mode automatic differentiation.
  - Visualization methods (`trace`, `visualize_tensor`) to display the computation graph.

## Neural Network Module

### Class `Module`

- **Purpose**: Base class for all neural network components.
- **Methods**:
  - `zero_grad()`: Resets gradients for all parameters.
  - `parameters()`: Returns a list of parameters (weights and biases).

### Class `Neuron`

- **Purpose**: Represents a single neuron in a neural network.
- **Attributes**:
  - `w`: Weights of the neuron.
  - `b`: Bias of the neuron.
  - `nonlin`: Non-linearity (activation function) applied to the neuron's output.
- **Methods**:
  - `__call__()`: Computes the output of the neuron given an input.

### Class `Layer`

- **Purpose**: Represents a layer of neurons.
- **Attributes**:
  - `Layer`: A list of neurons in the layer.
- **Methods**:
  - `__call__()`: Computes the output of the layer by passing inputs through each neuron.

### Class `MLP`

- **Purpose**: Represents a multi-layer perceptron (MLP) neural network.
- **Attributes**:
  - `layers`: A list of layers in the network.
- **Methods**:
  - `__call__()`: Performs a forward pass through the network.
  - `trace()`, `draw_dot()`, `visualize()`: Methods for visualizing the computation graph of the network.

## Usage

The library can be used to create and train simple neural networks. The `tensor` class allows for easy manipulation of scalar values with automatic differentiation, while the `nn` module provides a framework for building and training neural networks. Both tensor and MLP objects can be visualized for deeper understanding.

## Example

An example of using this library is provided in the `test.ipynb` file, where a simple MLP is trained on a dataset using backpropagation.