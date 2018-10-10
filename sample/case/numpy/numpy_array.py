import numpy as np

if __name__ == '__main__':
    _array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    shape = _array.shape
    print('形状:{}。 {}行，{}列'.format(shape, shape[0], shape[1]))
    print('Numpy中的结构:\n{}'.format(_array))
    print('Python 原始数据结构:\n{}'.format(_array.tolist()))

    _wrapper = np.array(_array)

    print('对numpy.array的数据结构再次使用numpy.array不会发生任何改变:\n{}, 类型:{}'.format(_wrapper, type(_wrapper)))
