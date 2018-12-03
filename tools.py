# coding: utf8


def standard_scale(x_train, x_test):
    """对训练集和测试集进行相同的标准化（scale）"""
    from sklearn import preprocessing
    preprocessor = preprocessing.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


def get_random_block_from_data(data, batch_size):
    """随机获取一个batch的数据"""
    import numpy as np
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


def unpickle(file):
    """反序列化：
    加载数据，解析成字典返回
    """
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


if __name__ == "__main__":
    data_batch_1 = unpickle("data/cifar10/data_batch_1")
    data_batch_2 = unpickle("data/cifar10/data_batch_2")
    data_batch_3 = unpickle("data/cifar10/data_batch_3")
    data_batch_4 = unpickle("data/cifar10/data_batch_4")
    data_batch_5 = unpickle("data/cifar10/data_batch_5")
    test_batch = unpickle("data/cifar10/test_batch")
