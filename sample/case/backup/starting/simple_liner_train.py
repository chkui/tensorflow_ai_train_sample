# -- coding: utf-8 --
# -*- coding:utf8 -*-

import tensorflow as tf


class LinerModel:
    def __init__(self, slope=.0, bias=.0, step=.1):
        self.slope = slope
        self.bias = bias
        self.train_step = step

    def compute(self, x):
        return  round(self.slope * x + self.bias, 4)

    def train(self, feature, label):
        """

        :param feature: 特征列表[2, 3]
        :param label: 标记列表[4, 5]
        :return:
        """
        self.__train_slope(feature, label)

    def __train_slope(self, point1, point2):
        x1 = point1[0]
        y1 = point1[1]

        x2 = point2[0]
        y2 = point2[1]

        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp

            temp = y1
            y1 = y2
            y2 = temp

        # 斜率旋转方向 -1 或 1
        ori = 1
        a = self.slope
        step = self.train_step
        stop = step/10

        # 偏移差数
        loss = 999.
        train_list = []
        count = 0

        while loss > stop:
            a = (a + ori * step)
            offset = (a * x1) - y1 - ((a * x2) - y2)
            ori = -1 if offset < 0 else 1
            train_list.append(offset)
            loss = abs(train_list[count]) if len(train_list) > 2 else 999.
            count += 1
            print('Train count={}, loss={}, Ori={}, slope={}'.format(count, loss, ori, a))
        self.slope = a
        print('Train Slope={}'.format(self.slope))
        self.bias = y1 - a * x1
        print('Train Bias={}'.format(self.bias))


def main(argv):
    point_1 = (1,1)
    point_2 = (2,2)
    point_3 = (3,3)

    liner = LinerModel()
    liner.train(point_1,point_2)
    print(liner.compute(point_3[0]))

    liner = LinerModel()
    liner.train((-1, 1),(-3, 6))
    print(liner.compute(5))

    liner.train((1, 9),(2, 6))
    print(liner.compute(5))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
