import numpy as np
import random
import matplotlib.pyplot as plt

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A function to perform a forward pass through the neural network. You can design your neural network however you want.
def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output

# TODO: Implement a simple evolutionayr learning algorithm to to optimize the weights of a feedforward neural network for the XOR problem.
# xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# xor_outputs = np.array([[0], [1], [1], [0]])