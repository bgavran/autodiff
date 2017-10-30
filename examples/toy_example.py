import numpy as np
import matplotlib.pyplot as plt
import automatic_differentiation as ad


class SynthGrad(ad.Module):
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.w = []
        for i in range(self.n_layers):
            w_initial = np.random.randn() * 0.01
            w = ad.Variable(w_initial, name="w" + str(i))
            self.w.append(w)

    def forward(self, x):
        out = x
        for i, w in enumerate(self.w):
            out = ad.Sigmoid(out * w)
        return out


def f(x_vals):
    return (x_vals > 2) * 1  # * 1 is for converting to integers


xlim = 7
sp = np.linspace(-xlim, xlim, 200)
all_x, all_b = np.meshgrid(sp, sp)
all_y = f(all_x)

x = ad.Variable(all_x, name="x")
y = ad.Variable(all_y, name="y")
b = ad.Variable(all_b, name="b")

p = ad.Sigmoid(x + b)
loss = ad.SquaredDifference(p, y)

synth_grad = SynthGrad(3)

b_grad = ad.grad(p, [b])[0]
b_synth = synth_grad(p)

synth_diff = ad.SquaredDifference(b_grad, b_synth)

optimizer = ad.Adam(len(synth_grad.w))
cost = ad.Einsum("ij->", synth_diff)
w_list_grads = ad.grad(cost, synth_grad.w)  # w_grads wrong dimension? problem because of some implicit broadcasts?
new_w_list = optimizer([i() for i in synth_grad.w], [i() for i in w_list_grads])

for w, new_w in zip(synth_grad.w, w_list_grads):
    w.value = new_w

    # cont = plt.contourf(x(), b(), synth_diff(), 50)
    # plt.colorbar(cont)
    # plt.pause(0.05)
    # plt.cla()
    # print("step", i)

# plt.show()
