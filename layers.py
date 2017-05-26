from computational_graph import Variable


class Dense:
    def __init__(self, inputs, n_outputs):
        """
        Creates a mapping from a list of inputs to the specified number of outputs.
        Uses a len(inputs)*n_outputs numbe of weights to achieve that.
        It represents a fully connected neural network layer
        
        :param inputs: a list of inputs
        :param n_outputs: number of outputs
        """
        self.n_weights = len(inputs) * n_outputs
        self.weights = [Variable(name="w") for _ in range(self.n_weights)]


