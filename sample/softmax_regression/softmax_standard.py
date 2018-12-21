import numpy as np
import random


class softmaxStandard:
    def __init__(self, features, labels):
        self.__features = features
        self.__labels = labels
        feature_num = features.shape[1]
        label_num = labels.shape[1]
        self.__weights = np.random.rand(label_num, feature_num) * 10


if __name__ == '__main__':
    feature_num = 100  # 特征个数
    classify_num = 20  # 样本分类
    total = 5000  # 样本数量
    range = 10  # 样本取值范围

    features = np.random.randint(range, size=(feature_num, total))
    labels = np.random.randint(1, size=(classify_num, total))

    for label in labels:
        label[random.randint(0, classify_num - 1)] = 1
    labels = np.matrix(labels)

    softmaxStandard(features, labels)
