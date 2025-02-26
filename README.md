# notes
Micrograd builds a mathematical expression with two inputs and a single output. It builds out the expression as an expression graph. During the forward pass, micrograd calculates the value of the output node. During the backward pass, micrograd initializes backpropagation at the output node: backward through the expression graph and recursively apply the chain rule to calculate the derivative of the output with respect to all the internal and input nodes. This represents how the input affects the output through the mathematical expression. Important note: micrograd is on scalar-level!

Neural networks are simple mathematical expressions that take the input data and weights of a neural network as an input, the output are the predictions or the loss function. So, Backpropagation is an algorithm to efficiently evaluate the gradient of a loss function with respect to the weights of a neural network. The weights can be iteratively tuned to minimize the loss function and improve the accuracy of a neural network. But important to know that backpropagation can be used to evaluate any kind of mathematical expression. 





Model mathematical expressions
- Forward pass: Get result y for expression
- Backward pass: Get gradient (if x goes up/down, what happens to y) for every x; backpropagation, derivative, chain rule...
nn is special form of mathematical expression
- Multi-layer perceptron: layers of neurons (weight, bias, activation)
- Forward pass: Get predictions for all input data.
- Backward pass: Get gradient for all weights and biases. Also gradient for input data, but this is given.
- Update: Tune nn parameters to minimize loss function (= sum of differences between predictions and gt). Give weights nudge in certain direction (!gradient gives direction of loss increase!)
- Iterate between forward pass, backward pass and update.

Important:
- If nudge (learning rate) is too high, the nn might overshoot the minimum loss + the loss might explode
- One step (epoch) trains on all input data.
- Zero grad: the gradients accumulate. So, don't forget to reset them every pass.

# Summary by Andrej:
- neural nets are very simple mathematical expressions that take the data and parameters as input.
- The expression is followed by a loss function. The loss funtion estimates the accuracy of the predictions (forward pass) on that input data. Loss function is low when the predictions match the gt.
- Backward the loss: use backpropagation to get the gradients. Gradients help to tune the parameters to minimize the loss. Iterate this in what's called gradient descent.

# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
