# -- coding: utf-8 --
# -*- coding:utf8 -*-

import tensorflow as tf

"""
线性模型：f(x) = ax + b
损失函数：loss(a, b) = 1/n∑(f(x) - y), n表示输入点的个数
梯度方向：∂loss(a, b)/∂a, ∂loss(a, b)/∂b, 分别对a、b求偏导
"""


class GradientLinerModel:

    def __init__(self, slope=.0, bias=.0, step=.1, momentum=.5):
        self.slope = slope
        self.bias = bias
        self.step = step
        self.momentum = momentum

    def compute(self, x):
        return round(self.slope * x + self.bias)

    def train(self, feature, label):
        for count in range(20):
            loss = self.__variance_loss__(feature, label)
            a = self.__variance_step_slope__(feature, label)
            b = self.__variance_step_bias__(feature, label)
            print('loss={}, a={}, b={} '.format(loss, a, b))

        # while abs(pre_loss - loss) > 0.0001:
        #     a = step * (partial_a(point1[x], point1[y], a, b) + partial_a(point2[x], point2[y], a, b)) - a
        #     b = step * (partial_b(point1[x], point1[y], a, b) + partial_b(point2[x], point2[y], a, b)) - b
        #     pre_loss = loss if 0 != count else 9999
        #     loss = compute_loss(point1[x], point1[y], a, b) + compute_loss(point2[x], point2[y], a, b)
        #     count += 1
        #     print('Train Count={}, Slope={}, Bias={}, Loss={}, pre-loss={} '.format(count, a, b, loss, pre_loss))

        self.slope = a
        self.bias = b

    def __liner_loss__(self, feature, label):
        count = len(feature)
        assert (count == len(label))
        loss = 0
        for index, x in enumerate(feature):
            loss += (self.slope * x + self.bias - label[index]) / count
        return loss

    def __variance_loss__(self, feature, label):
        length = len(feature)
        assert (length == len(label))
        loss = 0
        for index, x in enumerate(feature):
            y = label[index]
            loss += pow(self.slope * x + self.bias - y, 2) / length
        return loss

    def __variance_step_slope__(self, feature, label):
        length = len(feature)
        assert (length == len(label))
        derivative_sum = 0
        for index, x in enumerate(feature):
            y = label[index]
            derivative_sum += (2 * self.slope * pow(x, 2) + self.bias * x - x * y) / length

        v = derivative_sum * self.step + self.momentum * self.slope
        self.slope += v
        return self.slope

    def __variance_step_bias__(self, feature, label):
        length = len(feature)
        assert (length == len(label))
        derivative_sum = 0
        for index, x in enumerate(feature):
            y = label[index]
            derivative_sum += (2 * self.slope * x + self.bias - y) / length

        v = derivative_sum * self.step + self.momentum * self.bias
        self.bias += v
        return self.bias

    def __slope_step__(self, feature, label):
        count = len(feature)
        sum = 0
        #
        for x in feature:
            sum += x / count
        '''
        for循环是对线性方程求a的偏导=>f(X) = Xi/m m为输入特征的个数
        '''
        v = sum * self.step + self.momentum * self.slope
        '''
        v表示变更率
        '''
        self.slope += v
        return self.slope

    def __bias_step__(self, feature, label):
        self.bias = self.step + self.momentum * self.bias
        return self.bias


def main(argv):
    feature = [1, 2, 3]
    label = [1, 2, 3]

    model = GradientLinerModel()
    model.train(feature, label)

    print(model.compute(feature[2]))

    model2 = GradientLinerModel()
    model2.train([0, 1], [2, 2])
    print(model2.compute(0))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
