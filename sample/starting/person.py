# -- coding: utf-8 --

import numpy as np

LOOK = ['婉约', '大气']
EYE = ['小', '大']
MOUTH = ['薄', '厚']
NOSE = ['塌', '高']
FACE = ['圆', '方']


class Person:
    """
    个人样貌表述。
    将特征的描述设定为一个只包含0和1值的列表，0或者1表示不同的特征描述。
    [眼睛,嘴巴,鼻子,脸型]
    眼睛：0=小，1=大
    嘴巴：0=薄，1=厚
    鼻子：0=塌，1=高
    脸型：0=圆，1=方

    而结论也是通过0或者1来记录：0=婉约、1=大气

    分类原则
    1.凡是嘴厚脸型方的都归类为大气。而嘴薄脸型圆的归类为婉约。
    2.如果嘴小脸方或嘴大脸小则考察眼睛是否大，大=大气。
    3.如果眼睛小则考则鼻子， 高则大气，塌则婉约。

    """
    __Eye__ = 0
    __Mouth__ = 1
    __Nose__ = 2
    __Face__ = 3

    def __init__(self, name, feature):
        assert (4 == len(feature))
        self.name = name
        self.feature = feature
        self.look = None

    def classier(self):
        feature = self.feature
        if 1 == feature[self.__Mouth__] and 1 == feature[self.__Face__]:
            self.look = 1
        elif 0 == feature[self.__Mouth__] and 0 == feature[self.__Face__]:
            self.look = 0
        elif 1 == feature[self.__Eye__]:
            self.look = 1
        elif 1 == feature[self.__Nose__]:
            self.look = 1
        else:
            self.look = 0

        return self

    def result(self):
        return self.look

    def print(self):
        print('{}很{}'.format(self.name, LOOK[self.look]))


Person_Feature = {
    'Alice': [0, 0, 0, 0],
    'Bob': [1, 1, 1, 1],
    'Claire': [1, 0, 1, 0],
    'Deek': [1, 1, 1, 1],
    'Echo': [1, 0, 1, 1],
    'Frank': [1, 0, 0, 1],
    'Grace': [1, 1, 1, 0],
    'Kale': [0, 1, 1, 0],
    'Henry': [0, 1, 0, 0]
}
"""
    | Alice | 小 | 薄 | 塌 | 圆 | = 婉约 |
    | Bob | 大 | 厚 | 高 | 方 | = 大气 |
    | Claire | 大 | 薄 | 高 | 圆 | = 婉约 |
    | Deek | 大 | 厚 | 高 | 方 | = 大气 |
    | Echo | 大 | 薄 | 高 | 方 | = 大气 |
    | Frank | 大 | 薄 | 塌 | 方 | = 大气 |
    | Grace | 大 | 厚 | 高 | 圆 | = 大气 |
    | Kale | 小 | 厚 | 高 | 圆 | =大气 |
    | Henry | 小 | 厚 | 塌 | 圆 | =婉约 |
:return:
"""


def get_origin_feature():
    return Person_Feature


def get_look_dict():
    return LOOK


def main():
    for (name, feature) in Person_Feature.items():
        Person(name, feature).classier().print()

    print('随机测试：')

    # 生成一个10行4列的清单，每一行代表一个人的样貌特征
    random_feature = np.random.randint(low=0, high=2, size=(10, 4))

    for i, feature in enumerate(random_feature):
        name = '随机' + str(i)
        print('随机{}: 眼睛：{},嘴巴:{},鼻子:{},脸型:{}'.format(i, EYE[feature[0]], MOUTH[feature[1]], NOSE[feature[2]],
                                                     FACE[feature[3]]))
        Person(name, feature).classier().print()


def generate_simple(size=200):
    _features = np.random.randint(low=0, high=2, size=(size, 4))
    _label = [Person("random", feature).classier().result() for feature in _features]

    return _features, _label


if __name__ == '__main__':
    main()
    features, label = generate_simple()
    print('随机生成了{}行数据'.format(len(features)))
    print([features[i] for i in range(20)])
    print('………………')
