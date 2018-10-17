from sample.data.mnits_data_manager import MnitsDataManager
from sample.linear_regression.liner_model import LinerModel
import numpy as np


def one_batch(features, labels, start_pos, len):
    sub_features = features[start_pos:start_pos + len]
    out_features = []
    sub_labels = labels[start_pos:start_pos + len]
    for feature in sub_features:
        feature = feature.flatten()
        feature = feature * 4 / 1050
        out_features.append(feature)

    return out_features, labels[start_pos:start_pos + len]


def batch_group(features, labels, len=1000, total=50):
    """
    对要训练的数据进行分组
    :param features: 特征
    :param labels: 标签
    :param len: 一组的特征和标签个数
    :param total: 分组数量
    :return:
    """
    end_pos = features.shape[0] - len - 1
    start_pos = np.random.randint(low=0, high=end_pos, size=total)
    batch = []
    for pos in start_pos:
        batch_feature, batch_label = one_batch(features, labels, pos, len)
        batch.append((batch_feature, batch_label))
    return batch


def main():
    mn_data = MnitsDataManager()
    features, labels = mn_data.get_train()
    batchs = batch_group(features, labels)
    model = LinerModel()
    for batch in batchs:
        x = batch[0]
        y = batch[1]
        model.train(x, y)


if __name__ == '__main__':
    main()
