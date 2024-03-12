# Adapted from https://github.com/uber-research/PyTorch-NEAT
# and https://github.com/flowersteam/automated_discovery_of_lenia_patterns/tree/master/autodisc/autodisc/cppn

import torch
import torch.nn.functional as F

# Note: only delphineat_gauss, delphineat_sigmoid, tanh, sin activations constrain the outputs between [-1,1] => use this ones for CPPN generation


def delphineat_gauss_activation(z):
    """PyTorch implementation of gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT."""
    return 2.0 * torch.exp(-1 * (z * 2.5) ** 2) - 1


def delphineat_sigmoid_activation(z):
    """PyTorch implementation of sigmoidal activation function as defined in DelphiNEAT"""
    return 2.0 * (1.0 / (1.0 + torch.exp(-z * 5))) - 1


def tanh_activation(z):
    return torch.tanh(z * 2.5)


def sin_activation(z):
    return torch.sin(z * 5.0)


def sigmoid_activation(z):
    return torch.sigmoid(z * 5)


def gauss_activation(z):
    return torch.exp(-5.0 * z**2)


def identity_activation(z):
    return z


def relu_activation(z):
    return F.relu(z)


str_to_activation = {
    "delphineat_gauss": delphineat_gauss_activation,
    "delphineat_sigmoid": delphineat_sigmoid_activation,
    "tanh": tanh_activation,
    "sin": sin_activation,
    "sigmoid": sigmoid_activation,
    "gauss": gauss_activation,
    "identity": identity_activation,
    "relu": relu_activation,
}
