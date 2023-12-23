# Fourier Transform Neural Network

## Overview
The Fourier Transform Neural Network is a custom implementation of a neural network using TensorFlow. It employs a unique architecture involving multiple dense layers with custom activation functions, layer normalization, and dropout for regularization. This network is designed to handle tasks that may benefit from a Fourier transform-like approach in deep learning.

## Features
- **Custom Activation Function**: The network uses a custom-defined activation function, `tf.nn.relu` in this case, which can be replaced with any desired function.
- **Layer Normalization**: Each layer is normalized to stabilize the learning process and accelerate convergence.
- **Dropout**: Dropout layers are included for regularization to reduce overfitting.
- **Regularization**: L2 regularization is used to penalize complex models and avoid overfitting.

## Architecture
- The network consists of multiple dense layers, each followed by layer normalization and dropout.
- The final output is obtained through a dense layer with a sigmoid activation function.
- The architecture is flexible, allowing customization in the number of layers, units per layer, and regularization parameters.

## Callbacks
- **Custom Callback**: Monitors the start and end of each epoch.
- **Learning Rate Scheduler**: Adjusts the learning rate over epochs.
- **Early Stopping**: Stops training when validation loss does not improve.
- **Model Checkpoint**: Saves the best model based on validation loss.
- **TensorBoard**: For visualization and monitoring of training progress.

## Usage
- **Model Compilation**: The model is compiled with Adam optimizer and binary cross-entropy loss function.
- **Training**: The model is trained on random data for demonstration purposes. In practice, real datasets should be used.
- **Evaluation**: Post-training, the model's performance is evaluated on test data.


```python main.py```