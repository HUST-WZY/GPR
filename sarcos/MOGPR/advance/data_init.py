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

    # 获取公式中的各个值
    yt = data_y[1:, :]
    yt_1 = data_y[:-1, :]
    u = data_x[1:, :]
    y = yt - yt_1

    # 输出的维数
    n = y.shape[1]
    # 输入的维数
    m = u.shape[1]
    # 输入数据的个数
    len = y.shape[0]

    # 初始化 g 张量，g 是 len 个 n*m 的矩阵
    g = np.empty((n, m, len))

    # 填充 g 张量的值，这里根据 KKT 条件的公式
    for t in range(len):

        y_cur = y[t, :]
        u_cur = u[t, :]

        u_square_sum = 0

        for i in range(m):

            u_square_sum += u_cur[i] ** 2

        for i in range(n):

            yi = y_cur[i]

            for j in range(m):

                g[i, j, t] = yi * u_cur[j] / u_square_sum

    # 改变 g 的维度为 (len, n*m)， 便于后续训练
    g_squeeze = g.reshape(-1, g.shape[2]).T

    # n 此时为数据的总数
    n = g_squeeze.shape[0]
    train_n = (int)(floor(n * proportion))

    # 划分训练集和测试集，注意此时的输入为连续的时间点
    train_x = torch.linspace(1, train_n, train_n) - 1
    train_y = g_squeeze[:train_n, :]
    train_y = torch.from_numpy(train_y).contiguous().float()

    test_x = torch.linspace(train_n, n, n - train_n)
    test_y = g_squeeze[train_n:, :]
    test_y = torch.from_numpy(test_y).contiguous().float()

    return yt, yt_1, u, train_x.reshape(-1, 1), train_y, test_x.reshape(-1, 1), test_y


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)