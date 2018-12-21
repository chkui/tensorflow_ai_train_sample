import numpy as np
from numpy.linalg import inv


def plus():
    """
    矩阵加法
    :return:
    """
    # shape = (3,3)
    matrix = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 1 + matrix
    print('1\n + \n{}\n=\n{}'.format(matrix, 1 + matrix))
    print('=============================================')
    # matrix + 1
    print('\n{}\n + 1\n=\n{}'.format(matrix, matrix + 1))


def multiplication():
    W = np.matrix([[1, 2, 3]])
    X = np.matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 6, 7]])
    R = W * X.T
    print(R)
    print(R.T)
    print(R.shape)
    R_1 = X * W.T
    print(R_1)
    print(R_1.shape)


def division():
    print('矩阵除法操作。')
    print('矩阵除以一个元素：')
    A = np.matrix([[2, 3, 4], [7, 8, 9], [5, 6, 7]])
    B = np.matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    print('A矩阵：')
    print(A)

    print('B矩阵：')
    print(B)
    print('=============================================')
    print('A / 2 =')
    print(A / 2)
    """
    矩阵除以元素，矩阵中每个值除这个元素
    """
    print('=============================================')
    print('2 / A =')
    print(2 / A)
    """
    元素除以矩阵，矩阵中每个元素作为分母计算出结果
    """
    print('=============================================')
    print('矩阵相除：')
    print('B / A = ')
    print(B / A)
    '''
    在numpy中直接使用/符号计算，相当于对位元素相除，并不是真正的矩阵除法。
    矩阵除法常规的计算法则是B/A = B*Ai，（Ai是A的逆矩阵）
    '''

    print('-------')
    print('矩阵相除（通过逆矩阵计算）：')

    # matirx对象直接使用.I可以获得逆矩阵，也可以使用inv(matrix)函数
    Ai = A.I
    print('{}'.format(B * Ai))

    C = np.matrix([[1], [2], [3]])
    print(C)
    print('的逆矩阵：')
    print(C.I)
    print(C.I.T)


def exponent():
    A = np.matrix([[2, 3, 4], [7, 8, 9], [5, 6, 7]])
    print(np.exp(2))
    print(np.exp(3))
    print(np.exp(4))

    print(np.exp(A))


def main():
    # 矩阵转置
    _matrix_0 = np.matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
    print('原矩阵\n{}'.format(_matrix_0))
    print('转置矩阵\n{}'.format(np.transpose(_matrix_0)))

    print('矩阵所有元素求和\n{}'.format(np.sum(_matrix_0)))

    # 矩阵加减计算
    print('加法', np.matrix([[1, 2], [3, 4], [5, 6], [7, 8]]) + np.matrix([1, 2]))

    # 形状为(2, 4）的矩阵
    _matrix_1 = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    # 形状为(4, 2）的矩阵
    _matrix_2 = np.matrix([[1, 2], [3, 4], [5, 6], [7, 8]])

    # 矩阵与元素的计算
    print('元素与矩阵的加减5-A:\n{}'.format(5 - _matrix_1))
    print('矩阵与单个元素的计算可以直接使用*运算符:\n{}'.format(_matrix_1 * 2))

    # 矩阵乘法
    #
    _matrix_3 = _matrix_1 * _matrix_2
    """
    numpy老版本使用 np.matmul(_matrix_1, _matrix_2)
    """
    print('矩阵乘积预案算结果:\n{}. \n形状:{}'.format(_matrix_3, _matrix_3.shape))

    # 使用array类型可以实现对位元素相乘，而不是矩阵相乘
    print('对位元素相乘:\n{}'.format(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) * np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))
    '''
    直接使用*符号进行乘法实际上是数组相乘，会执行对位元素相乘，两个矩阵必须形状一样
    '''

    # 截取
    _matrix_4 = np.random.randint(low=1, high=10, size=(10, 10))

    print('被截取的矩阵:\n{}'.format(_matrix_4))
    print('截取第一行所有元素:{}'.format(_matrix_4[0, :]))
    print('截取第一列所有元素:{}'.format(_matrix_4[:, 0]))
    print('截取前3行的所有元素:\n{}'.format(_matrix_4[0:3, :]))
    print('截取前3列所有元素:\n{}'.format(_matrix_4[:, 0:3]))
    print('截取2、3行的第2～4列元素:\n{}'.format(_matrix_4[1:3, 1:3]))

    print('通过二维度数组的方式取固定值:\n{}'.format(_matrix_4[1][2]))


if __name__ == '__main__':
    plus()
    division()
    exponent()
    multiplication()

