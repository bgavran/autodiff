import numpy as np
import automatic_differentiation as ad

from examples.data import get_data


@ad.module_wrapper
def nn(x, w0):
    x_reshaped = ad.Reshape(x, ad.Shape(from_tuple=(-1, 28 * 28)))  # [batch_size, 28*28]
    return x_reshaped @ w0


@ad.module_wrapper
def softmax_ce_logits(labels, logits):
    return ad.Einsum("i->", ad.SoftmaxCEWithLogits(labels=labels, logits=logits))


@ad.module_wrapper
def optimizer(w, w_grad):
    lr = 0.001
    return w - lr * w_grad


w0 = ad.Variable(np.random.randn(28 * 28, 10) * 0.01, name="w0")

for i in range(2000):
    x_val, y_val = get_data(train=True)
    x, y = ad.Variable(x_val, name="x"), ad.Variable(y_val, name="y")

    logits = nn(x, w0)

    loss = softmax_ce_logits(y, logits)

    w0_grad = ad.Grad(loss, w0)
    new_w0 = optimizer(w0, w0_grad)

    w0 = ad.Variable(new_w0(), name="w0")

    if i % 100 == 0:
        print("Step:", i)
        print("Loss:", np.sum(loss()))
        print("-----------")

print("Testing...")
x, y = get_data(train=False)
x, y = ad.Variable(x, name="x"), ad.Variable(y, name="y")

test_network = ad.Softmax(nn(x, w0))
test_loss = softmax_ce_logits(y, nn(x, w0))

true = np.argmax(y(), -1)
pred = np.argmax(test_network(), -1)
print("True y:", true)
print("Predicted y:", pred)
print("Last batch accuracy:", np.mean(true == pred))
print("Loss:", np.sum(test_loss()))

new_w0.plot_comp_graph(view=True)
