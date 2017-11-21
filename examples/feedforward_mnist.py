import numpy as np
import autodiff as ad
import time

from examples.pytorch_data_loader import MNIST

input_size = 784
hidden_size = 32
out_size = 10
nn = ad.NN([input_size, hidden_size, out_size])
optimizer = ad.Adam(len(nn.w), lr=0.001)

batch_size = 100
mnist = MNIST(batch_size=batch_size)

start = time.time()
for epoch in range(2):
    for step, (images, labels) in enumerate(mnist.train_loader):
        x = ad.Variable(images.view(-1, 28 * 28).numpy(), name="images")
        y_true = ad.Variable(np.eye(10)[labels.numpy()], name="labels")

        y_logit = nn(x)

        sce = ad.SoftmaxCEWithLogits(labels=y_true, logits=y_logit)
        cost = ad.Einsum("i->", sce) / batch_size

        w_list_grads = ad.grad(cost, nn.w)

        new_w_list = optimizer([i() for i in nn.w], [i() for i in w_list_grads])
        optimizer.apply_new_weights(nn.w, new_w_list)

        if step % 100 == 0:
            text = "epoch {}, step {}, cost {:.2f}, grad norm {:.2f}, time {:.2f}"
            print(text.format(epoch, step, cost(), ad.FrobeniusNorm(*w_list_grads)(), time.time() - start))
            start = time.time()


def network_output(x):
    probs = ad.Softmax(nn(x))
    return probs()


def network_loss(x, y_true):
    return ad.Einsum("i->", ad.SoftmaxCEWithLogits(labels=y_true, logits=nn(x)))


print("Testing...")
total = 0
correct = 0
for images, labels in mnist.test_loader:
    x = ad.Variable(images.view(-1, 28 * 28).numpy(), name="images")
    y_true_num = ad.Variable(labels.numpy(), name="labels")
    predicted = np.argmax(network_output(x), -1)
    true = y_true_num()
    total += len(true)
    correct += (predicted == true).sum()

print("Accuracy on 10000 test images:", 100 * correct / total)
