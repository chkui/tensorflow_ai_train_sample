from sample.data.data_manager import DataManager


class MnitsDataManager(DataManager):
    """
    Mnits数据管理类
    """
    train_img_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    test_img_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    def get_train_features(self):
        """
        训练特征数据
        :return:
        """
        pass

    def get_train_label(self):
        """
        训练标签数据
        :return:
        """
        pass

    def get_text_features(self):
        """
        成功率测试数据
        :return:
        """
        pass

    def get_test_label(self):
        """
        成功率测试标签
        :return:
        """
        pass
