import numpy as np


class LinerModel:

    def __init__(self, train_count=300, dir=None):
        self.__w = None
        self.__b = None
        # 全局训练次数
        self.train_count = train_count
        # 中间值的存储位置
        self.__dir = dir

    def bias(self):
        """
        获取偏移量
        :return:
        """
        return self.__b

    def weight(self):
        """
        获取模型参数
        :return:
        """
        return self.__w

    def train(self, feature, label):
        x = np.matrix(feature)
        y = np.matrix(label)
        self.__init_params__(x, y)
        count = 0
        while count < self.train_count:
            self.__optimize_step__(x, y)
            loss = self.__loss__(x, y)
            count += 1
            print('第{}步, 方差:{}'.format(count, loss))
        return self

    def predict(self, features):
        """
        :param features:
        :return:
        """
        ori = self.__model__(features)
        return ori

    def __init_params__(self, x, y):
        """
        根据输入的特征和标记确定模型的参数形状（shape）
        """
        row = np.array(x, dtype=np.float64).shape[1]
        column = np.array(y, dtype=np.float64).shape[1]
        self.__w = np.matrix([[0 for i in range(column)] for j in range(row)], dtype=np.float64)
        self.__b = np.matrix([0 for i in range(column)], dtype=np.float64)
        return self

    def __model__(self, x):
        """
        运算模型。
        """
        return x * self.__w + self.__b

    def __loss__(self, x, y):
        """
        损失函数——方差：
        """
        d_values = self.__model__(x) - y
        d_values = np.array(d_values)
        d_values = d_values * d_values / x.shape[0]
        loss = np.sum(d_values)
        return loss

    def __optimize_step__(self, _X, _y, step=0.1):
        # 整个训练样本的规模
        m = len(_X)

        # 求样本空间Xj(k)的转置矩阵
        _TX = np.transpose(_X)

        # 计算LOSS
        loss = self.__model__(_X) - _y

        # 转置矩阵计算并得到一个和weight相同形状的矩阵，计算Δ(∂L/∂Wij)
        _dW = 2 * _TX * loss / m

        # 梯度更新权重矩阵的数值，即Wij = Wij - η × Δ
        self.__w = self.__w - step * _dW

        # 为计算偏移量的偏导数，创建一个(1, m)数值全为1/m的矩阵。
        _M = np.matrix([1 / m for i in range(_X.shape[0])],
                       dtype=np.float64)

        # 梯度更新Bi的数据，即即Bi = Bi - η × 2/m * ∑(LOSS)
        self.__b = self.__b - step * _M * loss

        return self
