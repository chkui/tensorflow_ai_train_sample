import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


# 交叉熵三维显示趋势图
def feature_entropy():
    label = [1, 0]
    o = np.linspace(1, 20, 50, dtype=float)
    x1 = []
    x2 = []
    h = []
    for _x1 in o:
        for _x2 in o:
            x1.append(_x1)
            x2.append(_x2)
            exp = np.exp([_x1, _x2])
            r = exp / np.sum(exp)
            h.append(np.sum(label * np.log(1 / r)))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Feature1')
    ax.set_ylabel('Feature2')
    ax.set_zlabel('Corss entropy')
    ax.scatter(x1, x2, h)
    plt.show()


# 值占比交叉熵趋势
def ratio_entropy():
    feature_num = random.randint(5, 10)
    pos = random.randint(1, feature_num) - 1
    label = [0 for n in range(feature_num)]
    label[pos] = 1

    def random_features(feature_num):
        return np.random.randint(200, size=(200, feature_num))

    loss = []
    h = []

    for i in range(1000):
        features = random_features(feature_num)
        flag = features[:, pos]
        loss.append(np.sum(flag) / np.sum(features))
        exp = np.exp(features)
        softmax = exp / np.sum(exp)
        h.append(np.sum(label * np.log(1 / softmax)) / 1000)

    plt.figure()
    plt.scatter(loss, h)
    plt.xlabel("loss")
    plt.ylabel("corss entropy")
    plt.show()


feature_entropy()
ratio_entropy()
