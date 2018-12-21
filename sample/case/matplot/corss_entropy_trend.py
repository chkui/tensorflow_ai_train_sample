import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


# 交叉熵三维显示趋势图
def feature_entropy():
    # 标签是2个分类，这里统计属于第一个分类时与q1、q2的数值关系
    label = np.matrix([1, 0]).T
    origin = np.linspace(0, 1, 100, dtype=float)
    q1 = []
    q2 = []
    h = []
    for _q1 in origin:
        _q2 = 1 - _q1
        q1.append(_q1)
        q2.append(_q2)
        h.append(np.log(np.matrix([_q1, _q2])) * label)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('Corss entropy')
    ax.scatter(q1, q2, h)
    plt.show()


# 值占比交叉熵趋势
def ratio_entropy():
    num = 20  # 样本分类
    total = 5000  # 样本个数
    range = 10 # 样本取值范围
    features = np.matrix(np.random.randint(range, size=(total, num)))
    labels = np.random.randint(1, size=(total, num))

    for label in labels:
        label[random.randint(0, num - 1)] = 1
    labels = np.matrix(labels)

    features_exp = np.exp(features)
    row_sums = features_exp*np.matrix(np.random.randint(low=1,high=2,size=(num, 1)))
    classify = features_exp/row_sums

    softmax_top = (classify * labels.T).diagonal()
    print(softmax_top.tolist()[0])
    loss = np.log(softmax_top)
    print(loss.tolist()[0])

    plt.figure()
    plt.scatter(loss.tolist()[0], softmax_top.tolist()[0])
    plt.xlabel("corss entropy")
    plt.ylabel("softmax highest")
    plt.show()

    # for i in range(total):
    #     feature = features[i];
    #     label = labels[i];
    #
    #
    # def random_features(feature_num):
    #     return np.random.randint(200, size=(200, feature_num))
    #
    # loss = []
    # h = []
    #
    # for i in range(1000):
    #     features = random_features(feature_num)
    #     flag = features[:, pos]
    #     loss.append(np.sum(flag) / np.sum(features))
    #     exp = np.exp(features)
    #     softmax = exp / np.sum(exp)
    #     h.append(np.sum(label * np.log(1 / softmax)) / 1000)
    #
    # plt.figure()
    # plt.scatter(loss, h)
    # plt.xlabel("loss")
    # plt.ylabel("corss entropy")
    # plt.show()


# feature_entropy()
ratio_entropy()
