import random
from math import floor

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


# 原始数据集的输出的维数
num_task = 7


def data_init(proportion):
    """
    初始化数据

    根据：
        y(t) = y(t - 1) + g(t)u(t)
    先给出 g(t) ，再划分训练测试集，最后返回

    :param proportion: 训练集和测试集的划分比例
    :return:
    """
    global num_task

    # 读取原始数据集
    train_data = pd.read_csv(r"../../res/sarcos_inv.csv", sep=',')
    test_data = pd.read_csv(r"../../res/sarcos_inv_test.csv", sep=',')

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # 竖着拼接，得到完整数据集
    data = np.concatenate((train_data, test_data), axis=0)

    data = data[:-5500, :]

    # 划分数据集的输入输出
    data_x = data[: , : -num_task]
    data_y = data[: , -num_task :]

    # 降维
    pca = PCA(n_components=0.9)
    data_x = pca.fit_transform(data_x)
    data_y = pca.fit_transform(data_y)

    # 降维后的输出的维度
    num_task = data_y.shape[1]

    # 横着拼接，得到完整数据集
    data = np.concatenate((data_x, data_y), axis=1)

    # 标准化
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    data = (data - mu) / sigma

    # 再次划分输入输出
    data_x = data[: , : -num_task]
    data_y = data[: , -num_task :]

    # 数据总数
    n = data_x.shape[0]
    train_n = (int)(floor(n * proportion))

    # 划分训练集和测试集
    train_x = data_x[: train_n, :]
    train_y = data_y[: train_n, :]

    test_x = data_x[train_n: , :]
    test_y = data_y[train_n: , :]

    train_x = torch.from_numpy(train_x).contiguous().float()
    train_y = torch.from_numpy(train_y).contiguous().float()
    test_x = torch.from_numpy(test_x).contiguous().float()
    test_y = torch.from_numpy(test_y).contiguous().float()

    return train_x, train_y, test_x, test_y


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)