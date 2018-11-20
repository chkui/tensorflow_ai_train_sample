import numpy as np
import matplotlib.pyplot as plt

x = origin_p = np.linspace(-5, 5, 1000, dtype=float)
exp = np.exp(x)
softmax = exp/np.sum(exp)

plt.figure()
plt.plot(x, softmax)
plt.xlabel("x")
plt.ylabel("softmax")
plt.show()

