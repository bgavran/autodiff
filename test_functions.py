import numpy as np


def zero_func(x1, x2):
    # if 1 < x1 < 1.5 and 1 < x2 < 1.5:
    #     return 5
    return 0


def if_func(x1, x2):
    if x1 >= 0:
        return x2
    return 0


def random(x1, x2):
    return np.random.randint(0, 5)


def larger_than(x1, x2):
    if x1 >= x2:
        return 1
    return 0


# regression
def test1(x, y):
    return 2 * x - y


def test2(x, y, param=0):
    if x > 1 and y < -1:
        return param
    return 0


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def composite(x1, x2):
    """
    Returns x2 if x1 is larger than x2
    Returns x2 if its smaller than x2
    :param x1:
    :param x2:
    :return:
    """
    return if_func(larger_than(x1, x2), x1)
