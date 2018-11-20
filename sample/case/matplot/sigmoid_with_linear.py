# 二维sigmoid函数图像

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    y = np.exp(x) / (1 + np.exp(x))
    return y


def linear(x):
    return .5 * x + 10


x = np.linspace(-6, 6, 50, dtype=float)
y = sigmoid(linear(x))

plt.plot(x, y)
plt.ylabel("Probability")
plt.xlabel("x")
plt.title("Sigmoid Function")
plt.show()
