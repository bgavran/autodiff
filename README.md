# Automatic differentiation 

This project represents my attempt to understand and implement neural networks from first principles.

Those first principles should encompass some of the new ideas in deep learning which challenge some of our assumptions of what neural networks really are.

### Current features
* Dynamic creation of computational graphs
* Dynamic differentiation of computational graph w.r.t. any variable
* Support for higher order gradients
* Extensible code (it's easy to add your own operations)
* Visualization of the computational graph
* Checkpointing

Disclaimer: This is a proof of concept, rather than a usable framework. It's not optimized and you can probably break any of the things above if you try. 
Foundations are there but I'm still trying understand how [good code](https://xkcd.com/844/) should look.

Everything is implemented only in Python 3.6 and numpy and is a heavy Work In Progress.

## Implementation details

An arbitrary neural network is implemented as a directed acyclic graph (DAG).

In this graph, a node represents an `Operation` and directed edges represent flow of information from one operation to another.

Operation can be any mathematical function at all. 
Incoming edges to an operation represent its domain and operation's outputs represent its codomain (its output).

Operations are closed under composition.
In other words, composition of outputs of two operations yields a new Operation.

Composition of operations can be abstracted and the inner operations hidden with the `CompositeOperation` class.

`Grad` is a CompositeOperation which takes an operation and returns a new DAG representing the gradient of that operation with respect to the provided variable.

Since Grad is a CompositeOperation, it's gradient can be taken in exactly the same way.


## What are really the first principles? 

These questions are some of my guidelines of deciding how this project should look like.

#### What it really means to _update parameters_ in a neural network?

Usually, the idea is: "lets compute forward pass, then compute gradients, put them through an optimizer and add the result to the parameters".

This seem to be the first principles for training neural networks.
This is how every backpropgation tutorial presents it, this is how I learned it and this is probably how most people learned it.

However, the [Synthetic Gradients paper](https://arxiv.org/abs/1608.05343) seems to challenge that idea.

What they did is they broke the feedback loop of updating the parameters into several smaller feedback loops, some of which __don't have any Gradient operations in them!__ And it still works! 
Obivously, the gradient information *is* used during the training, but it seems that a functional, efficient update can be performed with just an approximation of the Gradient.

This means that the core principles outlined above aren't really *core* principles and that there's something else going on.

#### What it means to _use an optimizer_ while training neural networks?

What was just a simple gradient has become a gradient multiplied by a learning rate.
What was just a gradient multipled with a learning rate has become Rmsprop.
Rmsprop became Adam and Adam became [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) and [Putting a Neural Network as the Optimizer](https://arxiv.org/abs/1606.04474).

This is a condensed history of designing optimizers.

The last two examples show us how it's possible to actually _learn_ the optimizer and that opens a whole new can of worms here.
All the questions we've been asking about our ML models can now be asked about the optimizers (and there's a lot of them!).

Obviously, the principle "oh we just take the gradient and change it a bit before adding it to the parameters" is not really a *core* principle after all and there is something much more deeper going on.
