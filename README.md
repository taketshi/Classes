# NN_torch
General implementation of dense NN with PyTorch

## Description
This is a implementation of a dense neural network with PyTorch with a certain degree of generality, to serve many purposes.

## Dependencies 
- import torch
- import torch.nn as nn
- import torch.optim as optim
- import torch.nn.functional as F

## Methods
\__init__():
- layer_sizes: list of integers which represent the number of neurons in each layer.
- activation_functions: list of string which represent the activation functions. They will be alternated with the linear layers. It does not add an activation function on the last layer.

forward_pass(): 
- input: pytorch tensor

train(): 
Trains the NN
- dataset: the dataset with features and target as pytorch tensors
- regularization: Determines the regularization. The keys 'l1', 'l2' or None implement the L1, L2 and no regularization, respectively
- criterion: the criterion for the loss function. The keys 'mse', 'cross_entropy' instruct to use the mean squared error and cross entropy, respectively
- optimizer: the way the descent will be made. For now, only accepts SGD.
- epochs: int representing the number of epochs we'll train
- lr: float representing the learning rate

accuracy():
The only non-general method. Implemented for specific project.

