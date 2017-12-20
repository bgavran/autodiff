import os
import numpy as np
import autodiff as ad
from examples.utils import TextLoader

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "sample_text_for_generation.txt")
text_loader = TextLoader(open(file_path).read())

batch_size = 8
input_size = text_loader.num_chars
hidden_size = 100
out_size = input_size
unroll_steps = 20

w = ad.Variable(np.random.randn(hidden_size, hidden_size) * 0.01, name="w")
u = ad.Variable(np.random.randn(input_size, hidden_size) * 0.01, name="u")
b_h = ad.Variable(np.zeros(hidden_size), name="bias_hidden")

v = ad.Variable(np.random.randn(hidden_size, out_size) * 0.01, name="v")
b_o = ad.Variable(np.zeros(out_size), name="bias_hidden")

params = [w, u, b_h, v, b_o]

optimizer = ad.Adam(len(params), lr=0.01)


def sample_text(seed_text="The", unroll_steps_after_seed=30, temperature=0.7):
    seed_text_onehot = text_loader.to_one_hot(text_loader.text_to_indices(seed_text))
    h = ad.Variable(np.zeros((1, hidden_size)), name="h")
    for seed_step in range(len(seed_text)):  # iterate through the seed text
        x = ad.Variable(seed_text_onehot[:, seed_step, :], name="x")
        h = ad.Tanh(h @ w + x @ u + b_h)
        logits = h @ v + b_o

    def sample_char(logits): # sample a character from the multinomial distribution
        next_char_onehot = np.random.multinomial(n=1, pvals=ad.Softmax(logits / temperature)()[0])
        next_char = text_loader.ind_to_char[np.argmax(next_char_onehot)]
        next_char_onehot = np.expand_dims(next_char_onehot, axis=0)
        return next_char_onehot, next_char

    for _ in range(unroll_steps_after_seed):  # autoregressively generate new characters
        next_char_onehot, next_char = sample_char(logits)
        seed_text += next_char
        x = ad.Variable(next_char_onehot, name="x")
        h = ad.Tanh(h @ w + x @ u + b_h)
        logits = h @ v + b_o
    return seed_text


for step in range(10000):
    x_batch_onehot = text_loader.to_one_hot(text_loader.next_batch(batch_size, seq_len=unroll_steps))
    h = ad.Variable(np.zeros((1, hidden_size)), name="h")
    costs = []
    for unroll_step in range(unroll_steps - 1):
        x = ad.Variable(x_batch_onehot[:, unroll_step, :], name="x")
        h = ad.Tanh(h @ w + x @ u + b_h)
        logits = h @ v + b_o
        y = ad.Variable(x_batch_onehot[:, unroll_step + 1, :])
        cost = ad.Einsum("i->", ad.SoftmaxCEWithLogits(labels=y, logits=logits))
        costs.append(cost)
    total_cost = ad.Add(*costs) / unroll_steps
    param_grads = ad.grad(total_cost, params)
    new_params = optimizer([i() for i in params], [i() for i in param_grads])
    optimizer.apply_new_weights(params, new_params)

    if step % 20 == 0:
        text = "step: {}, cost: {:.2f} \n------------------------------ \n {} \n------------------------------"
        print(text.format(step, float(total_cost()), sample_text()))
