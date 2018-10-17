# -- coding: utf-8 --
import numpy as np
from sample.starting.person import Person
from sample.starting.person import get_origin_feature
from sample.starting.person import get_look_dict


class LinerRegression:
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
        feature = np.matrix(feature)
        label = np.matrix(label)

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
        """

        :param features:
        :return:
        """
        ori = self.__model__(features)
        return ori

    def __init_params__(self, feature, label):
        """
        根据输入的特征和标记确定模型的参数形状（shape）
        :param feature: 输入特征，案例数据形状为（n, 4）
        :param label: 结论标签，案例数据形状为(n, 2)
        :return: this
        """
        np_feature = np.array(feature, dtype=np.float64)
        np_label = np.array(label, dtype=np.float64)
        row = np_feature.shape[1]
        column = np_label.shape[1]
        self.__weight = np.matrix([[0 for i in range(column)] for j in range(row)], dtype=np.float64)
        self.__bias = np.matrix([0 for i in range(column)], dtype=np.float64)
        return self

    def __model__(self, features):
        """
        运算模型。
        模型的线性表达式为： F(X) = Wij * Xj + Bi。但是实际运算我们都使用矩阵来执行。
        Wij Xj Bi 以及F(X)都是一个矩阵。根据案例的数据结构，用矩阵计算过程如下：

               Xj          ×      Wij      +      Bi       =        F(X)        i∈[0,2) j∈[0,4)

        [[X1, X2, X3, X4]    [[W11, W21]                       [[F(X)1, F(X)2]    样本1
         [X1, X2, X3, X4]     [W12, W22]                        [F(X)1, F(X)2]    样本2
         [X1, X2, X3, X4]  ×  [W13, W23]   +  [[B1, B2]]   =    [F(X)1, F(X)2]    样本3
         [X1, X2, X3, X4]     [W14, W24]]                       [F(X)1, F(X)2]    样本4
         ………………………………………]                                       …………………………………]    …………



        :param features: 特征数据
        :return:
        """
        np_feature = np.matrix(features)
        return np_feature * self.__weight + self.__bias

    def __loss__(self, features, labels):
        """
        损失函数——差异值的方差：

        LOSS = 1/m∑(Wij * Xj + Bi - Yi)^2 => LOSS = 1/m∑(F(X)ij - Yi)^2

        其中:
            i是标签分类值的个数，所以i∈[0, len(labels[0]))
            j是特征的个数，所以j∈[0, len(features[0]))
            求和公式和方差表示对输入的训练样本进行方差计算之后再全局求和。

        :param features:
        :param labels:
        :return:
        """
        d_values = self.__model__(features) - labels
        d_values = np.array(np.reshape(d_values, 400))[0]
        _len = len(d_values)
        loss = 0
        for e in d_values:
            loss += e * e / _len
        return loss

    def __optimize_step__(self, _X, _y, step=0.1):
        """
        优化器：损失函数__loss__对每一个权重参数求偏导。
        这里使用最基本的梯度递减方法来求每一个权重参数的递减方向，需要对每一个权重参数求偏导：

        已知：F(X)ij = Wij * Xj + Bi
        所以：样本计算值和真实数值的偏差是：LOSS = F(X)ij - Yi
        全部样本的方差为：L = 1/m∑(LOSS)^2。

        其中m为样本总量，求和公式对所有的特征计算数值和标签数值求方差。对权重Wij求偏导的过程如下：

        展开第一个样本数据的方差：

         L(0) =
            1/m × (W11 × X1 + W12 × X2 + …… + W1j × Xj + B1 - Y1)^2 +
            1/m × (W21 × X1 + W22 × X2 + …… + W2j × Xj + B2 - Y2)^2 +
            …………
            1/m × (Wi1 × X1 + Wi2 × X2 + …… + Wij × Xj + Bi - Yi)^2

        所以：

        1）对W11求偏导 =>
            ∂L/∂W11 = 2/m(W11 × X1 + W12 × X2 + …… + W1j × Xj + B1 - Y1) × X1 + 0 + ……

        2）对W12求偏导 =>
            ∂L/∂W12 = 2/m(W11 × X1 + W12 × X2 + …… + W1j × Xj + B1 - Y1) × X2 + 0 + ……

        2）对W21求偏导 =>
            ∂L/∂W12 = 2/m(W21 × X1 + W22 × X2 + …… + W1j × Xj + B1 - Y1) × X1 + 0 + ……

        3）对W)22求偏导 =>
            ∂L/∂W22 = 2/m(W21 × X1 + W22 × X2 + …… + W1j × Xj + B2 - Y2) × X2 + 0 + ……

        4）所以对Wij求偏导 =>
            ∂L/∂Wij = 2/m(Wi1 × X1 + Wi2 × X2 + …… + Wij × Xj + Bi - Yi) × Xj + 0 + ……

        5) 因此在整个样本空间的偏导公式为（对m个样本进行求和）：

            Δ(∂L/∂Wij) = 2/m∑(Wi1 × X1 + Wi2 × X2 + …… + Wij × Xj + Bi - Yi) × Xj
             => Δ = 2/m∑(k)∑i(j)(Wij × Xj + Bi - Yi) × Xj
             => Δ = 2/m∑(k)∑i(j)(LOSS) × Xj
            其中：
                ∑(k)表示求所有样本的合计：样本量=m => 下标k ∈ [0, m)
                ∑i(j)表示单个样本第i列的W对应j下标的合计值：i ∈ [0, len(labels[0])), j ∈ [0, len(features[0]))

        所以每一个权重参数的优化算法为：
            Wij = Wij - η × Δ（η是变动步长，也就是传入的参数step）

        同样的，对Bi求偏导数得到∂L/∂Bi = 2/m * ∑(LOSS)。

        在编码实现实际算法的时候，还是使用矩阵结合求和公式实现算法，这样通过numpy执行会高效很多，代码也相对简洁：
            观察Δ的表达式：其中的 ∑(k)∑i(j)(Wij × Xj + Bi - Yi) 就是运算模型减去标签值，所以可以先进行如下的矩阵计算：

               Xj          ×     Wij       +      Bi       -       Yi     =      LOSSi      i∈[0,2) j∈[0,4)

        [[X1, X2, X3, X4]    [[W11, W21]                       [[Y1, Y2]       [[L1, L2]        样本1
         [X1, X2, X3, X4]     [W12, W22]                        [Y1, Y2]        [L1, L2]        样本2
         [X1, X2, X3, X4]  ×  [W13, W23]   +  [[B1, B2]]   -    [Y1, Y2]   =    [L1, L2]        样本3
         [X1, X2, X3, X4]     [W14, W24]]                       [Y1, Y2]        [L1, L2]        样本4
         ………………………………………]                                       …………………]        …………………}

        矩阵LOSSi可以看做参与计算的常量，简写为L,表示是一个矩阵。下面以求W11为例详细说明求解过程：
            使用矩阵之后，上面的表达式替换为：

                2/m∑(k)∑i(j)L × Xjk,

                k表示样本空间的下标，k ∈ [0, m)

            对于单个Wij偏导而言，实际上是求矩阵L第i列与Xj的乘积，然后再求和。用X表示样本矩阵， Tx表示X的转置矩阵。 =>
                Tx                     ×         L               x

             (1)  (2) (3) (4) (m)
            [[X1, X1, X1, X1, ……]           [[L1, L2](1)
             [X2, X2, X2, X2, ……]      ×     [L1, L2](2)         ×       2/m
             [X3, X3, X3, X3, ……]            [L1, L2](3)
             [X4, X4, X4, X4, ……]}           [L1, L2](4)
                                                 …………………](m)

             对Bi求偏导结果与Xj无关，但是需要对结果集求和，所以创建一个(2, m)数值全为1(2/m)矩阵:
                      M            ×         L            ×

             (1)(2)(3)(4)(m)
            [[1, 1, 1, 1, ……]           [[L1, L2](1)
             [1, 1, 1, 1, ……]      ×     [L1, L2](2)      ×       2/m
                                         [L1, L2](3)
                                         [L1, L2](4)
                                         …………………](m)

            对Bi求偏导数
        :param features:
        :param labels:
        :param step:
        :return:
        """

        # 整个训练样本的规模
        m = len(_X)

        # 求样本空间Xj(k)的转置矩阵
        _TX = np.transpose(_X)

        # 计算LOSS
        loss = self.__model__(_X) - _y

        # 转置矩阵计算并得到一个和weight相同形状的矩阵，计算Δ(∂L/∂Wij)
        _dW = 2 * _TX * loss / m

        # 梯度更新权重矩阵的数值，即Wij = Wij - η × Δ
        self.__weight = self.__weight - step * _dW

        # 为计算偏移量的偏导数，创建一个(1, m)数值全为1/m的矩阵。
        _M = np.matrix([1 / m for i in range(_X.shape[0])],
                            dtype=np.float64)

        # 梯度更新Bi的数据，即即Bi = Bi - η × 2/m * ∑(LOSS)
        self.__bias = self.__bias - step * _M * loss

        return self


def normal_person_look(results):
    nors = abs(np.array(results) - 1)
    return [0 if item[0] < item[1] else 1 for item in nors]


if __name__ == '__main__':
    _features = np.random.randint(low=0, high=2, size=(200, 4))
    _label = [[1, 0] if 0 == Person("random", feature).classier().result() else [0, 1] for feature in _features]
    model = LinerRegression().train(_features, _label)
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
        print(
            '{}:预测=<{}>。实际=<{}>。'.format(keys[index], LOOK[nor], LOOK[Person(name, origin[name]).classier().result()]))

    print('========================')
    print('使用随机数据进行预测。')

    _features = np.random.randint(low=0, high=2, size=(10, 4))
    result = normal_person_look(model.predict(_features))
    for i, r in enumerate(result):
        v = _features[i]
        print('随机变量{}:预测=<{}>。实际=<{}>'.format(v, LOOK[r], LOOK[Person('', v).classier().result()]))
