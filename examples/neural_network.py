import numpy as np
import automatic_differentiation as ad

from examples.data import get_data

input_size = 784
hidden_size = 10
out_size = 10
nn = ad.NN([input_size, hidden_size, out_size])
optimizer = ad.Adam(len(nn.w))

batch_size = 32

for step in range(10000):
    x_val, y_val = get_data(train=True, batch_size=batch_size)
    x, y_true = ad.Variable(x_val, name="x"), ad.Variable(y_val, name="y")
    x = ad.Reshape(x, (-1, 28 * 28))  # [batch_size, 28*28]

    y_logit = nn(x)

    sce = ad.SoftmaxCEWithLogits(labels=y_true, logits=y_logit)
    cost = ad.Einsum("i->", sce) / batch_size

    w_list_grads = ad.grad(cost, nn.w)

    new_w_list = optimizer([i() for i in nn.w], [i() for i in w_list_grads])

    for w, new_w in zip(nn.w, new_w_list):
        w.value = new_w

    if step % 1000 == 0:
        text = "step {}, cost {:.2f}, grad norm {:.2f}"
        print(text.format(step, cost(), ad.FrobeniusNorm(*w_list_grads)()))

print("Testing...")
x, y = get_data(train=False, batch_size=100)
x, y_true = ad.Variable(x, name="x"), ad.Variable(y, name="y")
x = ad.Reshape(x, (-1, 28 * 28))  # [batch_size, 28*28]


def network_output(x):
    probs = ad.Softmax(nn(x))
    return probs()


def network_loss(x, y_true):
    return ad.Einsum("i->", ad.SoftmaxCEWithLogits(labels=y_true, logits=nn(x)))


true = np.argmax(y_true(), -1)
pred = np.argmax(network_output(x), -1)
print("True y:", true)
print("Predicted y:", pred)
print("Last batch accuracy:", np.mean(true == pred))
print("Loss:", network_loss(x, y_true)())

# sum(w_list_grads).plot_comp_graph()
