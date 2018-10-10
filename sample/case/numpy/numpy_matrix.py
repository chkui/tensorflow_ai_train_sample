import numpy as np

if __name__ == '__main__':
    # 矩阵加减计算
    print('加法', np.array([[1, 2], [3, 4], [5, 6], [7, 8]]) + np.array([1, 2]))

    # 形状为(2, 4）的矩阵
    _matrix_1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # 形状为(4, 2）的矩阵
    _matrix_2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # 矩阵与元素的计算
    print('元素与矩阵的加减:\n{}'.format(5 - _matrix_1))
    print('矩阵与单个元素的计算可以直接使用*运算符:\n{}'.format(_matrix_1 * 2))


    # 矩阵乘法
    _matrix_3 = np.matmul(_matrix_1, _matrix_2)
    print('矩阵乘积预案算结果:\n{}. \n形状:{}'.format(_matrix_3, _matrix_3.shape))

    # 对位元素相乘
    print('对位元素相乘:\n{}'.format(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) * np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))
    '''
    直接使用*符号进行乘法实际上是数组相乘，会执行对位元素相乘，两个矩阵必须形状一样
    '''

    # 矩阵截取
    _matrix_4 = np.random.randint(low=1, high=10, size=(10, 10))
    print('被截取的矩阵:\n{}'.format(_matrix_4))
    print('截取第一行所有元素:{}'.format(_matrix_4[0, :]))
    print('截取第一列所有元素:{}'.format(_matrix_4[:, 0]))
    print('截取前3行的所有元素:\n{}'.format(_matrix_4[0:3, :]))
    print('截取前3列所有元素:\n{}'.format(_matrix_4[:, 0:3]))
    print('截取2、3行的第2～4列元素:\n{}'.format(_matrix_4[1:3, 1:3]))

    print('通过二维度数组的方式取固定值:\n{}'.format(_matrix_4[1][2]))
