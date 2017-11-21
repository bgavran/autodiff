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

Disclaimer: This is a proof of concept, rather than a usable framework. It's not optimized and you can probably break any of the things above if you try. 
Foundations are there but I'm still trying understand how [good code](https://xkcd.com/844/) should look.

I welcome any feedback on the project.

Everything is implemented only in Python 3.6 and numpy and is a heavy Work In Progress.

## Usage

Clone the git repo and run the pip installer:

~~~~
git clone git@github.com:bgavran/autodiff.git
cd autodiff
pip install .
~~~~

Example usage:

~~~~
>>> import numpy as np
>>> import autodiff as ad
>>> x = ad.Variable(np.random.randn(2, 3), name="x")
>>> w = ad.Variable(np.random.randn(3, 5), name="w")
>>> y = x @ w
>>> y
< autodiff.core.ops.Einsum object at 0x7fe8870faef0>
>>> y()
array([[ 0.5323577 , -0.34353342,  1.33145506, -0.29360625, -1.56014675],
       [ 0.38069571,  0.40819971, -0.66564586,  0.16482348,  1.15585051]])
>>> w_grad = ad.grad(y, [w])[0]
>>> w_grad
< autodiff.core.ops.Add object at 0x7fe886f5cdd8>
>>> w_grad()
array([[-0.26209637, -0.26209637, -0.26209637, -0.26209637, -0.26209637],
       [ 0.61349261,  0.61349261,  0.61349261,  0.61349261,  0.61349261],
       [ 1.10694982,  1.10694982,  1.10694982,  1.10694982,  1.10694982]])
~~~~


## Implementation details

Autodiff is just a thin wrapper around numpy and uses it as the numerical kernel.

An arbitrary neural network is implemented as a directed acyclic graph (DAG).

In this graph, a node represents some mathematical operation and directed edges into the node represent arguments of the operation.

Composition of outputs of two nodes yields a new Node - operations are closed under composition.

Node can directly implement its mathematical operation - class `Primitive` - or it can use other, existing nodes - class `Module`.
Modules allow abstracting compositions of operations as just another operation. Modules also allow arbitrarily deep hierarchical nesting of operations.

__The cool part:__

Backpropagation is a higher-order function that maps one computational graph to another - the graph of the derivative.
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

Visualization idea and inspiration taken from [HIPS autograd](https://github.com/HIPS/autograd).
