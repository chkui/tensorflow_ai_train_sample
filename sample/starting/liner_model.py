# -- coding: utf-8 --
import numpy as np
from sample.starting.person import Person
from sample.starting.person import get_origin_feature
from sample.starting.person import get_look_dict


class LinerModel:
    """
    针对 sample.starting.person 的数据案例，原始开发的线性学习模型：

    F(X) = Wij * Xj + Bi

    根据案例数据的结构，数据的矩阵结构为：
    Aij的形状：(4,2)：
    Bi的形状：(1,2)。
    输入特征变量的形状：(n, 4)
    结果集F(X)的形状：(n, 2)，n表示输入的训练或预测数据个数。

    线性模型类的对外接口功能主要是train和predict，前者用于根据输入样本进行训练，后者用于进行数据预测。
    除此之外，bias和weight
    """
    # 默认训练次数
    def_train_step = 300

    def __init__(self):

        # 模型参数，即F(X) = Wij * Xj + Bi中的Wij，也被称为权重参数。
        self.__weight = None
        # 偏移量，Bi，在线性模型中用于对坐标轴进行修正
        self.__bias = None

    def bias(self):
        """
        获取偏移量
        :return:
        """
        return self.__bias

    def weight(self):
        """
        获取模型参数
        :return:
        """
        return self.__weight

    def train(self, feature, label):
        """
        数据训练接口
        :param feature: 特征数组，结构为(n, 4)。
        :param label: 标签数组，结构为(n, 2)。
        :return:
        """

        # 初始化参数和偏移量的矩阵形状
        self.__init_params__(feature, label)
        count = 0

        # 使用所有的输入数据进行训练， 默认训练300次
        while count < self.def_train_step:
            self.__optimize_step__(feature, label)
            loss = self.__loss__(feature, label)
            count += 1
            print('第{}步, 方差:{}'.format(count, loss))
        return self

    def predict(self, features):
        ori = self.__model__(features)
        return ori

    def __init_params__(self, feature, label):
        """
        根据输入的特征和标记确定模型的参数形状（shape）
        :param feature:
        :param label:
        :return:
        """
        np_feature = np.array(feature, dtype=np.float64)
        np_label = np.array(label, dtype=np.float64)
        row = np_feature.shape[1]
        column = np_label.shape[1]
        self.__weight = np.array([[0 for i in range(column)] for j in range(row)], dtype=np.float64)
        self.__bias = np.array([0 for i in range(column)], dtype=np.float64)
        return self

    def __model__(self, features):
        np_feature = np.array(features)
        return np.matmul(np_feature, self.__weight) + self.__bias

    def __optimize_step__(self, features, labels, step=0.1):

        size = len(features)
        base = self.__model__(features) - labels
        w = self.__weight
        row = w.shape[0]
        column = w.shape[1]

        # 对Aij求偏导数
        for i in range(row):
            for j in range(column):
                r = base[:, j]
                x = features[:, i]
                s = r * x / size
                sum = 0
                for e in s.tolist():
                    sum += e
                w[i][j] = w[i][j] - step * sum
        self.__weight = w
        base = self.__model__(features) - labels
        b = self.__bias
        row = b.shape[0]
        for i in range(row):
            r = base[:, i]
            sum = 0
            for e in r.tolist():
                sum += e / size
            b[i] = b[i] - step * sum
        self.__bias = b
        return self

    def __loss__(self, features, labels):
        d_values = self.__model__(features) - labels
        d_values = d_values.flatten()
        _len = len(d_values)
        loss = 0
        for e in d_values:
            loss += e * e / _len
        return loss


def normal_person_look(results):
    nors = abs(results - 1)
    return [0 if item[0] < item[1] else 1 for item in nors]


if __name__ == '__main__':
    _features = np.random.randint(low=0, high=2, size=(200, 4))
    _label = [[1, 0] if 0 == Person("random", feature).classier().result() else [0, 1] for feature in _features]
    model = LinerModel().train(_features, _label)
    print('训练完毕！')
    print('权重:\n{}\n偏移量:\n{}'.format(model.weight(), model.bias()))

    print('========================')
    print('使用预定义数据进行预测。')

    origin = get_origin_feature()
    result = model.predict([v for (k, v) in origin.items()])
    normals = normal_person_look(result)

    LOOK = get_look_dict()
    keys = list(origin.keys())

    for index, nor in enumerate(normals):
        name = keys[index]
        print('{}:\n预测属于“{}”类型。'.format(keys[index], LOOK[nor]))
        print('实际为属于[{}]类型。'.format(LOOK[Person(name, origin[name]).classier().result()]))

    print('========================')
    print('使用随机数据进行预测。')

    _features = np.random.randint(low=0, high=2, size=(10, 4))
    result = normal_person_look(model.predict(_features))
    for i, r in enumerate(result):
        v = _features[i]
        print('随机变量{}:\n预测属于“{}”类型。'.format(v, LOOK[r]))
        print('实际为属于[{}]类型。'.format(LOOK[Person('', v).classier().result()]))


