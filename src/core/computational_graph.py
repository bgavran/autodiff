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
        from core.ops import Add
        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from core.ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return -self.__add__(other)

    def __mul__(self, other):
        from core.ops import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        from core.ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from core.ops import MatMul
        return MatMul(other, self)

    def __truediv__(self, other):
        from core.ops import Recipr
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

    def f(self, input_dict):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class Operation(Node):
    @Node._constant_wrapper
    def __init__(self, children, name=""):
        super().__init__(name)
        self.children = children
        self.last_df = None
        self.all_names = self.check_names()

    def check_names(self):
        """
        Return names of all the nodes in the graph
        :return: list of all the node names in the tree below (and including) this node
        """
        temp_names = [self.name]
        for child in self.children:
            temp_names.extend(child.all_names)
        return temp_names

    def compute_derivatives(self, input_dict, grad=None):
        """
        Computes the derivatives for specific inputs for the *entire* computational graph, w.r.t. all variables
        Derivatives w.r.t. a *certain variable* are accumulated in the other method method, gradient()

        :param input_dict: dictionary of input variables
        :param grad:
        :return:
        """
        # self.check_names()

        if grad is None:
            grad = np.ones_like(self.f(input_dict))
        # Computing the derivative with respect to each of the inputs
        self.last_df = [self.df(input_dict, wrt=child.name, grad=grad) for child in self.children]

        # Making each of the inputs do the same
        for i, child in enumerate(self.children):
            if isinstance(child, Operation):
                child.compute_derivatives(input_dict, grad=self.last_df[i])

    def accumulate_all_gradients(self, wrt=""):
        """
        Accumulates all the gradients w.r.t. a specific variable
        
        :param wrt: function returns the gradient with respect to the variable whose name is provided here
        :return: gradient in the shape of the wrt variable
        """

        grad_sum = 0
        for i, child in enumerate(self.children):
            if wrt == child.name:
                grad_sum += self.last_df[i]
            else:
                grad_sum += child.accumulate_all_gradients(wrt=wrt)
        return grad_sum

    def accumulate_all_gradients_in_list(self, input_dict, wrt_list):
        """

        :param input_dict: dictionary where keys are strings of inputs to comp graph (w and x) and values are either a
                            2d array of all possible values (in case of weights) or just a single float (in case of x)
        :param wrt_list: a list where the gradient is computed for each item separately
        :return: a list o gradients of length len(wrt)
        """
        self.compute_derivatives(input_dict)
        return [self.accumulate_all_gradients(wrt=var) for var in wrt_list]

    def f(self, input_dict):
        raise NotImplementedError()

    def df(self, input_dict, wrt="", grad=None):
        """
        Should always return a tensor in the shape of one of its input variables

        :param input_dict:
        :param wrt:
        :param grad:
        :return:
        """
        raise NotImplementedError()


class Constant(Operation):
    def __init__(self, value, name=None):
        assert isinstance(value, np.ndarray) or isinstance(value, numbers.Number)
        if name is None:
            name = str(value)
        super().__init__([], name)
        self.value = value
        self.all_names = [self.name]

    def f(self, input_dict):
        return self.value

    def df(self, input_dict, wrt="", grad=None):
        return 0


class Variable(Operation):
    def __init__(self, name=""):
        """

        Right now, all variables are placeholders? There are no predefined values, all Variables need their values to be
        fed in
        :param name:
        """
        super().__init__([], name)
        self.all_names = [self.name]  # names of all the children nodes
        self.last_f = None

    def f(self, input_dict):
        try:
            return input_dict[self.name]
        except KeyError:
            raise KeyError("Must input value for variable", self.name)

    def df(self, input_dict, wrt="", grad=None):
        if wrt == self.name:
            return grad * np.ones_like(self(input_dict))
        else:
            return np.zeros_like(self(input_dict))


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

    def df(self, input_dict, wrt="", grad=None):
        return self.out.df(input_dict, wrt=wrt)

    def accumulate_all_gradients(self, wrt=""):
        return self.out.accumulate_all_gradients(wrt)

    def compute_derivatives(self, input_dict, grad=None):
        self.out.compute_derivatives(input_dict, grad)
