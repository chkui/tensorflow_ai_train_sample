# 数据的模型拟合图

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


class Module():

    def __init__(self):
        self.step = .1
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0

    def __weight__(self, gpa, gre):
        return self.w0 + self.w1 * gpa + self.w2 * gre

    def __sigmoidf__(self, t):
        return np.exp(t) / (1 + np.exp(t))

    def __sigmoidb__(self, t):
        return 1 / (1 + np.exp(t))

    def __loss__(self, features, labels):
        loss = 0
        for i in range(len(features)):
            feature = features[i]
            x1 = feature[0]
            x2 = feature[1]
            y = labels[i]
            loss = loss + y * np.log(self.probability_admission(x1, x2)) + (1 - y) * np.log(
                self.probability_unadmission(x1, x2))

        loss = loss / len(features)
        return loss

    def __update_weight__(self, features, labels):
        regular = 0
        for i in range(len(features)):
            feature = features[i]
            x1 = feature[0]
            x2 = feature[1]
            y = labels[i]
            probability = self.probability_admission(x1, x2)
            regular = regular + y - probability

        length = len(features)
        _w1 = np.sum(features[:, 0])
        _w2 = np.sum(features[:, 1])

        self.w0 = self.w0 - self.step * regular
        self.w1 = self.w1 - self.step * regular * _w1 / length
        self.w2 = self.w2 - self.step * regular * _w2 / length

    def probability_admission(self, gpa, gre):
        return self.__sigmoidf__(self.__weight__(gpa, gre))

    def probability_unadmission(self, gpa, gre):
        return self.__sigmoidb__(self.__weight__(gpa, gre))

    def train(self, features, labels):
        count = 300
        for i in range(count):
            self.__update_weight__(features, labels)
            loss = self.__loss__(features, labels)
            print('{}, w0:{}.w1:{}.w2:{}.loss:{}'.format(i, self.w0, self.w1, self.w2, loss))

    def predict(self, gpa, gre):
        return self.probability_admission(gpa, gre), self.probability_unadmission(gpa, gre)


w0 = 0
w1 = 0.01
w2 = 0.01


def liner_3d(x, y):
    return w0 + w1 * x + w2 * y


def sigmoid_2d(t):
    return np.exp(t) / (1 + np.exp(t))


if __name__ == '__main__':
    admissions = pd.read_csv('admissions.csv').get_values()
    admissions.dtype = 'float'
    total_number = len(admissions)

    flags = list(range(total_number))
    train_number = int(total_number * .7)
    validate_number = total_number - train_number

    train_data = []
    for i in range(train_number):
        flag_pos = random.randint(0, len(flags) - 1)
        flag = flags.pop(flag_pos)
        data = admissions[flag]
        train_data.append(data)
    train_data = np.array(train_data)

    validate_data = []
    for flag in flags:
        validate_data.append(admissions[flag])
    validate_data = np.array(validate_data)

    features = train_data[:, 1:3]
    labels = train_data[:, 0]

    module = Module()
    module.train(features, labels)

    gpa = admissions[:, 1]
    gre = admissions[:, 2]

    liner = liner_3d(gpa, gre)

    z = sigmoid_2d(liner)

    # 创建 3D 图形对象
    fig = plt.figure()
    # plt.xlabel('a')
    ax = Axes3D(fig)
    ax.set_xlabel('gpa')
    ax.set_xlabel('gre')
    ax.set_xlabel('sigmoid')

    # 绘制线型图
    ax.scatter(gpa, gre, z)

    # 显示图
    plt.show()
