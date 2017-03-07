from functools import lru_cache
import numbers


class Node:
    id = 0

    def __init__(self, name):
        Node.id += 1
        if name == "":
            self.name = str(Node.id)
        else:
            self.name = name
        self.all_names = None

    def __str__(self):
        return self.name

    def __mul__(self, other):
        from ops import Mul
        if isinstance(other, Node):
            return Mul([self, other])
        elif isinstance(other, numbers.Number):
            return Mul([Variable(other), self])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        from ops import Add
        if isinstance(other, Node):
            return Add([self, other])
        elif isinstance(other, numbers.Number):
            return Add([Variable(other), self])

    def __radd__(self, other):
        return self.__add__(other)


class Variable(Node):
    def __init__(self, value, name=""):
        super().__init__(name)
        self.value = value
        self.all_names = [self.name]  # names of all the children nodes

    def gradient(self, wrt=""):
        if wrt == self.name:
            return 1
        return 0

    def __call__(self, *args, **kwargs):
        return self.value


class Operation(Node):
    def __init__(self, children, name=""):
        super().__init__(name)
        self.children = children
        self.last_grad = None
        self.all_names = self.check_names()

    @lru_cache(maxsize=None)
    def f(self):
        raise NotImplementedError()

    @lru_cache(maxsize=None)
    def df(self, wrt=""):
        raise NotImplementedError()

    def check_names(self):
        """
        Avoiding two nodes with the same name in the graph
        :return: list of all the node names in the tree below (and including) this node
        :raises: ValueError if there are two equal names
        """
        temp_names = [self.name]
        for child in self.children:
            temp_names.extend(child.all_names)
        if len(set(temp_names)) != len(temp_names):
            raise ValueError("Names of nodes have to be unique!", temp_names)
        return temp_names

    def compute_gradient(self):
        self.check_names()
        # Computing the derivative with respect to each of the inputs
        self.last_grad = [self.df(wrt=child.name) for child in self.children]

        # Making each of the inputs do the same
        for child in self.children:
            if isinstance(child, Operation):
                child.compute_gradient()

    def gradient(self, wrt=""):
        grad_sum = 0
        for grad, child in zip(self.last_grad, self.children):
            grad_sum += grad * child.gradient(wrt=wrt)
        return grad_sum

    def __call__(self, *args, **kwargs):
        return self.f()


class CompositeOperation(Operation):
    def __init__(self, children, name=""):
        super().__init__(children, name)
        self.out = None
        self.graph()

    def graph(self):
        raise NotImplementedError()

    def f(self):
        self.out.f()

    def df(self, wrt=""):
        self.out.df(wrt=wrt)

    def gradient(self, wrt=""):
        return self.out.gradient(wrt)

    def compute_gradient(self):
        self.out.compute_gradient()
