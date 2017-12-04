# Automatic differentiation 

![](/assets/higher_order.png)

This project represents my attempt to understand and implement neural networks from first principles.

Those first principles should encompass some of the new ideas in deep learning which challenge some of our assumptions of what neural networks really are.

Neural network training seems to, at the bare minimum, require optimization of composed functions using gradient information.
Therefore, this project consist of implementation of an automated way to do that: automatic differentiation of arbitrary computational graphs.

### Current features
* Dynamic creation of computational graphs
* Dynamic differentiation of computational graph w.r.t. any variable
* Support for higher order gradients
* Support for higher order tensors
* Extensible code (it's easy to add your own operations)
* Visualization of the computational graph
* Numerical checks
* Checkpointing

Check out the `examples` directory to see it in action!

Disclaimer: This is a proof of concept, rather than a usable framework. It's not optimized and you can probably break any of the things above if you try. 
Foundations are there but I'm still trying understand how [good code](https://xkcd.com/844/) should look.

I welcome any feedback on the project.

Everything is implemented only in Python 3.6 and numpy.

## Usage

Clone the git repo and run the pip installer:

~~~~
git clone https://github.com/bgavran/autodiff.git
cd autodiff
pip install .
~~~~

Example usage:

```python
>>> import autodiff as ad
>>> x = ad.Variable(3, name="x")
>>> y = ad.Variable(4, name="y")
>>> z = x * y + ad.Exp(x + 3)
>>> z
< autodiff.core.ops.Add object at 0x7fe91284f080>
>>> z()
array(415.4287934927351)
>>> x_grad = ad.grad(z, [x])[0]
>>> x_grad
< autodiff.core.ops.Add object at 0x7fe9125ecc18>
>>> x_grad()
array(407.4287934927351)
>>> z.plot_comp_graph()

Plotting...
```

![](/assets/comp_graph.png)


Example with numpy arrays:

```python
>>> import numpy as np
>>> import autodiff as ad
>>> x = ad.Variable(np.random.randn(2, 3), name="x")
>>> w = ad.Variable(np.random.randn(3, 5), name="w")
>>> y = x @ w
>>> y()
array([[-0.43207813, -0.89955503,  1.08968335, -1.93750297,  1.68805722],
       [-0.33109636,  0.165466  ,  1.17712353,  0.32994463,  0.7632344 ]])
>>> w_grad, x_grad = ad.grad(y, [w, x])
>>> w_grad()
array([[ 0.49907665,  0.49907665,  0.49907665,  0.49907665,  0.49907665],
       [ 0.07703569,  0.07703569,  0.07703569,  0.07703569,  0.07703569],
       [-1.811908  , -1.811908  , -1.811908  , -1.811908  , -1.811908  ]])
```


## Implementation details

Autodiff is just a thin wrapper around numpy and uses it as the numerical kernel.

An arbitrary neural network is implemented as a directed acyclic graph (DAG).

In this graph, a node represents some mathematical operation and directed edges into the node represent arguments of the operation.

Composition of outputs of two nodes yields a new Node - operations are closed under composition.

All operations extend Node and implement its forward and backward pass.
Nodes can be grouped together in a `Module` - a structure which allows easy reusability of composition of nodes.
Modules can be nested in an arbitrarily deep hierarchy.

To get the gradient of a function - you use `grad`.

__The cool part:__

`grad` is a higher-order function that maps one computational graph to another - the graph of the derivative.
It doesn't use the information about derivatives at all, it's a function only of the graph structure!
Since it dynamically creates computational graphs, you can take the gradient of the gradient at no extra cost!
This is in contrast as to how backpropagation is usually presented and its inner mechanisms obscured with the many linear algebra operations. 

## Side note - what are really the first principles of learning mechanisms? 

These questions are some of my guidelines of deciding how this project should look like.

#### What it really means to _update parameters_ in a neural network?

Usually, the idea is: "lets compute forward pass, then compute gradients, put them through an optimizer and add the result to the parameters".

This seem to be the first principles for training neural networks.
This is how every backpropgation tutorial presents it, this is how I learned it and this is probably how most people learned it.

However, the [Synthetic Gradients paper](https://arxiv.org/abs/1608.05343) seems to challenge that idea.

What they did is they broke the feedback loop of updating the parameters into several smaller feedback loops, some of which __don't have any Gradient operations in them!__ And it still works! 
Obivously, the gradient information *is* used during the training, but it seems that a functional, efficient update can be performed with just an approximation of the gradient.

This means that the core principles outlined above aren't really *core* principles and that there's something else going on.

#### What it means to _use an optimizer_ while training neural networks?

What was just a simple gradient has become a gradient multiplied by a learning rate.
What was just a gradient multipled with a learning rate has become Rmsprop.
Rmsprop became Adam and Adam became [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) and [Putting a Neural Network as the Optimizer](https://arxiv.org/abs/1606.04474).

This is a condensed history of designing optimizers.

The last two examples show us how it's possible to actually _learn_ the optimizer and that opens a whole new can of worms here.
All the questions we've been asking about our ML models can now be asked about the optimizers (and there's a lot of them!).

Obviously, the principle "oh we just take the gradient and change it a bit before adding it to the parameters" is not really a *core* principle after all and there is something much more deeper going on.

---

Tanh visualization idea and inspiration taken from [HIPS autograd](https://github.com/HIPS/autograd).
