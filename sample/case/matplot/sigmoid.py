# 二维sigmoid函数图像

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    y = np.exp(x) / (1 + np.exp(x))
    return y


x = np.linspace(-6, 6, 50, dtype=float)
y = sigmoid(x)

plt.plot(x, y)
plt.ylabel("Probability")
plt.xlabel("x")
plt.title("Sigmoid Function")
plt.show()
