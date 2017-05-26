from functools import lru_cache
import numbers
import numpy as np


class Node:
    id = 0

    def __init__(self, name):
        self.id = Node.id
        Node.id += 1
        if name == "":
            self.name = "id_" + str(self.id)
        else:
            self.name = name
        self.all_names = None

    def __str__(self):
        return self.name

    def __add__(self, other):
        from ops import Add
        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        from ops import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from ops import Recipr
        return self.__mul__(Recipr(other))

    def __rtruediv__(self, other):
        return 1 / self.__truediv__(other)

    @staticmethod
    def _constant_wrapper(init_function):
        """
        Decorator for the Node class which wraps any normal number into a Constant
        :param init_function:
        :return: 
        """

        def wrap_all_children(self, children, name):
            for i, child in enumerate(children):
                if isinstance(child, numbers.Number):
                    children[i] = Constant(child)
            return init_function(self, children, name)

        return wrap_all_children


class Constant(Node):
    def __init__(self, value, name=None):
        assert isinstance(value, np.ndarray) or isinstance(value, numbers.Number)
        if name is None:
            name = "Constant"
        super().__init__(name)
        self.value = value
        self.all_names = [self.name]

    def f(self, input_dict):
        return self.value

    def gradient(self, wrt=""):
        return 0

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Variable(Node):
    def __init__(self, name=""):
        # All variables are, by default, placeholders?
        super().__init__(name)
        self.all_names = [self.name]  # names of all the children nodes

    def f(self, input_dict):
        try:
            return input_dict[self.name]
        except KeyError:
            raise KeyError("Must input value for variable", self.name)

    def gradient(self, wrt=""):
        return wrt == self.name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Operation(Node):
    @Node._constant_wrapper
    def __init__(self, children, name=""):
        super().__init__(name)
        self.children = children
        self.last_grad = None
        self.all_names = self.check_names()

    @lru_cache(maxsize=None)
    def f(self, input_dict):
        raise NotImplementedError()

    @lru_cache(maxsize=None)
    def df(self, input_dict, wrt=""):
        raise NotImplementedError()

    def check_names(self):
        """
        Return names of all the nodes in the graph
        :return: list of all the node names in the tree below (and including) this node
        """
        temp_names = [self.name]
        for child in self.children:
            temp_names.extend(child.all_names)
        return temp_names

    def compute_derivatives(self, input_dict):
        """
        Computes the derivatives for specific inputs for the entire computational graph, w.r.t. all variables
        Derivatives w.r.t. a certain variable are accumulated in the other method method, gradient()

        :param input_dict: dictionary of input variables
        :return:
        """
        # self.check_names()
        # Computing the derivative with respect to each of the inputs
        self.last_grad = [self.df(input_dict, wrt=child.name) for child in self.children]

        # Making each of the inputs do the same
        for child in self.children:
            if isinstance(child, Operation):
                child.compute_derivatives(input_dict)

    def gradient(self, wrt=""):
        """
        Accumulates all the gradients w.r.t. a specific variable
        
        :param wrt: function returns the gradient with respect to the variable whose name is provided here
        :return: 
        """
        grad_sum = 0
        for grad, child in zip(self.last_grad, self.children):
            grad_sum += grad * child.gradient(wrt=wrt)
        return grad_sum

    def gradient_list(self, input_dict, wrt):
        """

        :param input_dict: dictionary where keys are strings of inputs to comp graph (w and x) and values are either a
                            2d array of all possible values (in case of weights) or just a single float (in case of x)
        :param wrt: a list where the gradient is computed for each item separately
        :return: a list o gradients (in array form) of length len(wrt)
        """
        self.compute_derivatives(input_dict)
        return np.array([self.gradient(wrt=var) for var in wrt])

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class CompositeOperation(Operation):
    def __init__(self, children, name=""):
        """
        The inheriting class should call the graph() method in its own __init__()
        :param children: 
        :param name: 
        """
        super().__init__(children, name)
        self.out = None

    def graph(self):
        """
        Graph should set the self.out parameter
        :return: 
        """
        raise NotImplementedError()

    def f(self, input_dict):
        return self.out.f(input_dict)

    def df(self, input_dict, wrt=""):
        return self.out.df(input_dict, wrt=wrt)

    def gradient(self, wrt=""):
        return self.out.gradient(wrt)

    def compute_derivatives(self, input_dict):
        self.out.compute_derivatives(input_dict)
