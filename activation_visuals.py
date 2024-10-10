# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:03:54 2024

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return 2 / (1 + np.exp(-2*x)) - 1

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Generate input range
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")

plt.subplot(1, 4, 2)
plt.plot(x, relu(x))
plt.title("ReLU")

plt.subplot(1, 4, 3)
plt.plot(x, tanh(x))
plt.title("Tanh")

plt.subplot(1, 4, 4)
plt.plot(x, softmax(x))
plt.title("Softmax")

plt.tight_layout()
plt.show()