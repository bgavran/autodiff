# Automatic differentiation 

This project represents my attempt to understand and implement what I consider to be basic principles of learning mechanisms.

As I learn I revise my definitions this project changes in a similar manner.

Currently the idea is to create some sort of decetralized multi-agent system where neural network modules could be updated asynchronously and yield complex behaviour.

The idea is to create a framework where concepts such as synthetic gradients and neural networks as optimizers aren't a hack, but are a feature.

### Current features
* Dynamic creation of computational graphs
* Dynamic differentiation of computational graph w.r.t. any variable
* Support for higher order gradients
* Extensible code (it's easy to add your own operations)

Disclaimer: You can probably break any of those things above if you try hard enough. 
Foundations are there but I'm still trying understand how [good code](https://xkcd.com/844/) should look.

Everything is implemented only in Python 3.6 and numpy.

## Technical details

An arbitrary neural network is implemented as a directed acyclic graph (DAG).

In this graph, a node  represents an `Operation` and directed edges represent flow of information from one Operation to another.

Incoming edges to an Operation represent its domain and Operation's outputs represent its codomain (its output).

Composition of operations is also an Operation.
Composition of operations can be abstracted and it's innner operations hidden with the `CompositeOperation` class.

`Grad` is a CompositeOperation which takes a node and returns a new DAG representing the Gradient of that node with respect to a provided variable.

Since Grad is a CompositeOperation, it's graph can be taken without much hassle.




