# Lab11
## XOR Problem Solver using Evolutionary Learning Algorithm

This Python script demonstrates the use of an Evolutionary Learning Algorithm to optimize the weights of a feedforward neural network for solving the XOR problem.


## Implementation Details

### Activation Function

The script uses a sigmoid activation function:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
### Neural Network
The neural network contains two layers:

First layer: 2 input neurons, 3 hidden neurons
Second layer: 3 hidden neurons, 1 output neuron
Weights are represented as a flat NumPy array, which is reshaped when passed to the predict function:
```python
def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output
```

### TODO: Evolutionary Learning Algorithm
The user is expected to implement an Evolutionary Learning Algorithm to optimize the weights of the neural network for solving the XOR problem. The inputs and outputs for the XOR problem are provided:
```python
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])
```

Once the algorithm is implemented, the script can be used to solve the XOR problem using the optimized weights.
