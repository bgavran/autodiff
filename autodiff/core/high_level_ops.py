from .node import *
from .ops import *
from .reshape import *


class Module:
    def forward(self, *args, **kwargs):
        with add_context(type(self).__name__):
            return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NN(Module):
    def __init__(self, sizes):
        if len(sizes) < 2:
            raise ValueError("Sizes represent the layer sizes and needs to have at least two values for one weight"
                             " matrix to exist!")
        self.sizes = sizes
        self.w = []
        for i, (inp, out) in enumerate(zip(self.sizes, self.sizes[1:])):
            w_initial = np.random.randn(inp + 1, out) * 0.01
            w = Variable(w_initial, name="w" + str(i))
            self.w.append(w)

    def _forward(self, x):
        first_dim = x.shape[0]
        bias = Variable(np.ones((first_dim, 1)), name="bias")
        for i, w in enumerate(self.w):
            x = Concat(x, bias, axis=1)
            x = x @ w
            if i < len(self.w) - 1:
                x = ReLU(x)
        return x


class Optimizer(Module):
    @staticmethod
    def apply_new_weights(weight_list, new_weight_values):
        for w, new_w_value in zip(weight_list, new_weight_values):
            w.value = new_w_value

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()


class SGDOptimizer(Optimizer):
    def __init__(self, lr=0.001):
        # Doesn't have a state, which means it isn't tied to a specific problem
        self.lr = lr

    def _forward(self, params_list, grads_list):
        return [self.fn(param, grad) for param, grad in zip(params_list, grads_list)]

    def fn(self, param, grad):
        return param - self.lr * grad


# TODO test these optimization methods!
class Momentum(Optimizer):
    """
    Accumulates gradient change through time steps
    """

    def __init__(self, num_params, lr=0.001, gamma=0.8):
        # Each parameter has it's own momentum so this is tied to a specific model
        self.lr = lr
        self.gamma = gamma
        self.v = [0 for _ in range(num_params)]

    def _forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.v[i] = self.fn(self.v[i], param, grad)
        return params_list

    def fn(self, prev_m, param, grad):
        new_m = self.gamma * prev_m + grad
        return param - self.lr * new_m, new_m


class Adagrad(Optimizer):
    """
    Different learning rate for each parameter!
    The lr value is the same, but each parameter divides it by its own sqrt of something

    Implicitly takes into account how much the gradients are changing - 2nd order moment
    """

    def __init__(self, num_params, lr=0.01):
        self.lr = lr
        self.run_avg = [0 for _ in range(num_params)]

    def _forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.run_avg[i] = self.fn(self.run_avg[i], param, grad)
        return params_list

    def fn(self, prev_avg, param, grad):
        new_avg = prev_avg + grad ** 2
        new_param = param - self.lr / np.sqrt(new_avg + 1e-8) * grad
        return new_param, new_avg


class Adam(Optimizer):
    eps = 1e-8

    def __init__(self, num_params, lr=0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.m = [0 for _ in range(num_params)]
        self.v = [0 for _ in range(num_params)]
        self.step = 0

    def _forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.m[i], self.v[i] = self.fn(self.step, self.m[i], self.v[i], param, grad)
        self.step += 1
        return params_list

    def fn(self, step, prev_m, prev_v, param, grad):
        new_m = self.b1 * prev_m + (1 - self.b1) * grad
        new_v = self.b2 * prev_v + (1 - self.b2) * grad ** 2
        a = self.lr * np.sqrt(1 - self.b2 ** step) / (1 - self.b1 ** step + Adam.eps)
        new_param = param - a * new_m / (np.sqrt(new_v) + Adam.eps)
        return new_param, new_m, new_v


class NesterovMomentum(Optimizer):
    """
    Doesn't just use the gradient, but requires it's evaluation on a point that depends on the self.v parameter?
    """

    # TODO implement the fn method

    def __init__(self, num_params, lr=0.001, gamma=0.8):
        # Each parameter has it's own momentum so this is tied to a specific model
        self.lr = lr
        self.gamma = gamma
        self.v = [0 for _ in range(num_params)]

    def _forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.v[i] = self.fn(self.v[i], param, grad)
        return params_list

# class Convolution(Module):
#     def _forward(self, x, w):
#         shp = tuple(np.subtract(x.shape, w.shape[:-1]) + 1) + (w.shape[-1],)
#         s = w.shape[:-1] + shp[:-1]
#
#         strides = x().strides * 2  # TODO x is evaluated here!
#         subm = AsStrided(x, shape=s, strides=strides)
#
#         w_let = letters_from_tuple(w.shape)
#         op_str = w_let + "," + w_let[:-1] + "...->" + w_let[-1] + "..."
#         return Einsum(op_str, w, subm)
#
#
# Convolution = Convolution()
