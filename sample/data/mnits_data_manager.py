from sample.data.data_manager import DataManager
import tensorflow as tf
import numpy as np
import gzip
import struct
import tempfile


class MnitsDataManager(DataManager):
    """
    Mnits数据管理类。官方网站对Mnist的文件格式有详细的介绍。
    images(feature)文件的结构：
        0000~0003: 4字节(32位)   魔法数字，固定为0x00000803(2051)
        0004~0007: 4字节(32位)   用于记录当前样本中图片的总个数
        0008~0011: 4字节(32位)   标记样本中一张图片有多少行，即纵向有多少个像素点
        0012~0015: 4字节(32位)   标记样本中一张图片有多少列，即横向有多少个像素点
        0016     : 1字节(32位)   像素点的灰度值[0,255]
        0017     : 1字节(32位)   像素点的灰度值[0,255]
        ………………
        xxxx     : 1字节(32位)

    label文件结构：
        0000~0003: 4字节(32位)   魔法数字，固定为0x00000801(2049)
        0004~0007: 4字节(32位)   用于记录当前样本中图片的总个数
        0016     : 1字节(32位)   单个图片的标签[0,9]
        0017     : 1字节(32位)   单个图片的标签[0,9]
        ………………
        xxxx     : 1字节(32位)

    一张图片的结构是28×28像素，所以一张图片的像素个数是784
    """

    # 训练特征
    train_img_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'

    # 训练标签
    train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'

    # 测试标签
    test_img_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'

    # 测试特征
    test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    __url__ = {'train_img': (train_img_url, 2051),
               'train_label': (train_label_url, 2049),
               'test_img': (test_img_url, 2051),
               'test_label': (test_label_url, 2049)}

    def __init__(self):
        self.__file_path__ = {}

    def __check_file__(self, local, magic):
        with gzip.GzipFile(local) as g_file:
            buffer = g_file.read(4)
            i = int.from_bytes(buffer, 'big')
            if i != magic:
                raise ValueError('Magic number Error:{}'.format(local))

    def __read_zip_bytes__(self, byte_stream, inpos, offset=None, fmt='>I'):
        if offset:  byte_stream.read(offset)
        byte = byte_stream.read(inpos)
        return struct.unpack(fmt, byte)[0]

    def __download__(self):
        """
        下载MNITS数据
        """
        for (key, param) in self.__url__.items():
            url = param[0]
            zip_name = url.split('/')[-1]
            local = tf.keras.utils.get_file(zip_name, url)
            self.__file_path__[key] = local

        return self

    def __dual_images__(self, key):
        """
        对MNIST的训练或测试特征进行处理：假设样本空间中图片的总数为M。
        :param key:
        :return:
        """
        if 'train_img' != key and 'test_img' != key:
            raise ValueError('File Flag Error:{}'.format(key))

        local = self.__file_path__[key]
        npz = local + '.npz'
        features = None
        if tf.gfile.Exists(npz):
            features = self.__read_npz__(npz)
        else:
            with gzip.GzipFile(local) as g_file:
                def read(size):
                    return int.from_bytes(g_file.read(size), 'big')

                magic = read(4)
                if magic != self.__url__[key][1]:
                    raise ValueError('Magic number Error:{}'.format(local))

                # 图片个数 = M
                total_num = read(4)
                # 单个图片的行数据
                row_num = read(4)
                # 单个图片的列数
                column_num = read(4)
                # 一张图片的像素总
                pixel_num = row_num * column_num

                # 所有图片的列表
                images = []

                def read_image():
                    img = []
                    column = None
                    count = 0
                    img_buffer = g_file.read(pixel_num)
                    for j in range(pixel_num):
                        if 0 == count:
                            column = []
                            img.append(column)
                        column.append(img_buffer[j])
                        count = (count + 1) % column_num
                    return np.matrix(img)

                for i in range(total_num):
                    images.append(read_image())
                self.__write_npz__(npz, images)
                features = np.array(images)
        return features

    def __dual_labels__(self, key):
        local = self.__file_path__[key]
        npz = local + '.npz'
        labels = None
        if tf.gfile.Exists(npz):
            labels = self.__read_npz__(npz)
        else:
            with gzip.GzipFile(local) as g_file:
                def read(size):
                    return int.from_bytes(g_file.read(size), 'big')

                magic = read(4)
                if magic != self.__url__[key][1]:
                    raise ValueError('Magic number Error:{}'.format(local))

                # 标签个数 = M
                total_num = read(4)
                labels = []
                for i in range(total_num):
                    label = [0 for j in range(10)]
                    number = read(1)
                    label[number] = 1
                    labels.append(label)

                self.__write_npz__(npz, labels)
                labels = np.array(labels)
        return labels

    def __write_npz__(self, file_path, data):
        f = tempfile.NamedTemporaryFile()
        np.savez_compressed(f, data=data)
        tf.gfile.Copy(f.name, file_path)

    def __read_npz__(self, file_path):
        with np.load(file_path) as npzData:
            return npzData['data']

    def get_train(self):
        self.__download__()
        features = self.__dual_images__('train_img')
        labels = self.__dual_labels__('train_label')
        """
        训练特征、与标签数据数据
        :return: feature, label
        """
        return features, labels

    def get_text(self, number=None):
        """
        获取测试标签与测试数据
        :return: feature, label
        """
        self.__download__()
        features = self.__dual_images__('test_img')
        labels = self.__dual_labels__('test_label')

        return features, labels


if __name__ == '__main__':
    manager = MnitsDataManager()
    feature, label = manager.get_train()
    print('Train Feature shape:{}'.format(feature.shape))
    print('Train Label shape:{}'.format(label.shape))

    feature, label = manager.get_text()
    print('Test Feature shape:{}'.format(feature.shape))
    print('Test Label shape:{}'.format(label.shape))
