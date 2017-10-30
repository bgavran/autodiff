import numpy as np
import automatic_differentiation as ad


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NN(Module):
    def __init__(self, sizes):
        self.sizes = sizes
        self.w = []
        for i, (inp, out) in enumerate(zip(self.sizes, self.sizes[1:])):
            w_initial = np.random.randn(inp + 1, out)
            w = ad.Variable(w_initial, name="w" + str(i))
            self.w.append(w)

    def forward(self, x):
        first_dim = x.shape[0]
        bias = ad.Variable(np.ones((first_dim, 1)), name="bias")
        for i, w in enumerate(self.w):
            x = ad.Concat(x, bias, axis=1)
            x = x @ w
            if i < len(self.w) - 1:
                x = ad.ReLU(x)
        return x


class SGDOptimizer(Module):
    def __init__(self, lr=0.001):
        # Doesn't have a state, which means it isn't tied to a specific problem
        self.lr = lr

    def forward(self, params_list, grads_list):
        return [self.fn(param, grad) for param, grad in zip(params_list, grads_list)]

    def fn(self, param, grad):
        return param - self.lr * grad


# TODO test these optimization methods!
class Momentum(Module):
    """
    Accumulates gradient change through time steps
    """

    def __init__(self, num_params, lr=0.001, gamma=0.8):
        # Each parameter has it's own momentum so this is tied to a specific model
        self.lr = lr
        self.gamma = gamma
        self.v = [0 for _ in range(num_params)]

    def forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.v[i] = self.fn(self.v[i], param, grad)
        return params_list

    def fn(self, prev_m, param, grad):
        new_m = self.gamma * prev_m + grad
        return param - self.lr * new_m, new_m


class Adagrad(Module):
    """
    Different learning rate for each parameter!
    The lr value is the same, but each parameter divides it by its own sqrt of something

    Implicitly takes into account how much the gradients are changing - 2nd order moment
    """

    def __init__(self, num_params, lr=0.01):
        self.lr = lr
        self.run_avg = [0 for _ in range(num_params)]

    def forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.run_avg[i] = self.fn(self.run_avg[i], param, grad)
        return params_list

    def fn(self, prev_avg, param, grad):
        new_avg = prev_avg + grad ** 2
        new_param = param - self.lr / np.sqrt(new_avg + 1e-8) * grad
        return new_param, new_avg


class Adam(Module):
    eps = 1e-8

    def __init__(self, num_params, lr=0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.m = [0 for _ in range(num_params)]
        self.v = [0 for _ in range(num_params)]
        self.step = 0

    def forward(self, params_list, grads_list):
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


class NesterovMomentum(Module):
    """
    Doesn't just use the gradient, but requires it's evaluation on a point that depends on the self.v parameter?
    """

    # TODO implement the fn method

    def __init__(self, num_params, lr=0.001, gamma=0.8):
        # Each parameter has it's own momentum so this is tied to a specific model
        self.lr = lr
        self.gamma = gamma
        self.v = [0 for _ in range(num_params)]

    def forward(self, params_list, grads_list):
        for i, (param, grad) in enumerate(zip(params_list, grads_list)):
            params_list[i], self.v[i] = self.fn(self.v[i], param, grad)
        return params_list


# TODO better way to implement these two functions?
class Tanh(Module):
    def forward(self, x):
        val = ad.Exp(-2 * x)
        return (1 - val) / (1 + val)


Tanh = Tanh()


class SquaredDifference(Module):
    def forward(self, x, y):
        diff = x - y
        return diff ** 2


SquaredDifference = SquaredDifference()
