# notes
Micrograd builds a mathematical expression with two inputs and a single output (as an expression graph*). During the forward pass, micrograd calculates the value of the output node. During the backward pass, micrograd initializes backpropagation at the output node: backward through the expression graph and recursively apply the chain rule to calculate the derivative of the output with respect to all the internal and input nodes. This represents how the input affects the output through the mathematical expression. Important note: micrograd can only build scalar-values mathematical expressions!

*A mathematical expression can be represented as an expression graph, with variable and operation nodes. For example, for the following expression: a+b=c, then a and b are the children nodes; c is the parent node; plus is an operation node, which connects the children and parent nodes. 

During backward pass, the gradient (the derivative of the output node with respect to some node*) is calculated for all the intermediate nodes starting at the output node. The gradient for the output node is obviously one. To calculate the derivative for the intermediate nodes, the chain rule is very important: If a node z depends on a node y, which itself depends on a node x, then z depends on x as well, via the intermediate variable y. In this case, the chain rule is expressed as dz/dx = (dz/dy) * (dy/dx) (Wikipedia). So, imagine this as a backpropagation-signal (carrying the information of the derivative of the output node with respect to all the intermediate nodes) flowing thhrough the graph. A plus node basically routes the derivative to its children nodes. 

*A derivative represents the response of a function when the input is slightly altered (=the slope of the function). The mathematical formula for the derivative is: (f2 - f1)/h, where f1 is the output with input a and f2 is the output with a+h. When a function has multiple inputs, the derivative can be calculated with respect to all the inputs. 

Micrograd can easily be build out in PyTorch. To perform the forward pass: lace the scalars in tensors, set require_grad to true (default is False), perform the mathematical expression. To  perform the backward pass: call .backward() on the output. Every scalar has a .grad() and a .data() like in micrograd.

Neural networks are simple mathematical expressions that take the input data and weights of a neural network as an input, the output are the predictions or the loss function. So, Backpropagation is an algorithm to efficiently evaluate the gradient of a loss function with respect to the weights of a neural network. The weights can be iteratively tuned to minimize the loss function and improve the accuracy of a neural network. But important to know that backpropagation can be used to evaluate any kind of mathematical expression. 

A neural network is build out of (input, hidden and output) layers of fully-connected neurons. A neuron has inputs, weights and a bias (The weights represent the strength of each input. The bias represents the trigger-happiness of a neuron). The output of a neuron is the activation function applied to the dot-product of the weights and the inputs with added bias. 

During backward pass, the gradient is calculated for the output of the neuron with respect to the weights of the neuron. 



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
