import numpy as np


def pred_cal(yt, yt_1, u, mean_pred):

    """
    根据
        y(t) = y(t - 1) + g(t)u(t)
    由预测的 g 值，计算预测的输出
    :param yt:
    :param yt_1:
    :param u:
    :param mean_pred: g 的预测值
    :return:
    """

    # 输入维数
    num_input = u.shape[1]
    # 输出维数
    num_task = yt.shape[1]

    # 测试数据的个数
    test_n = mean_pred.shape[0]
    # 训练数据的个数
    train_n = u.shape[0] - test_n

    # 调整 g 的预测值的维度
    g_pred = mean_pred.cpu().T.reshape(num_task, num_input, test_n).numpy()

    # 计算预测的输出值
    y_pred = np.empty((test_n, num_task))
    for t in range(test_n):
        # 这里一定要注意，yt_1 和 u 的索引从谁开始
        y_pred[t, :] = yt_1[train_n + t, :] + np.matmul(g_pred[:, :, t], u[train_n + t, :])

    # 实际的输出值
    y_true = yt[-test_n:]

    error = y_true - y_pred

    MAE_y = np.zeros([num_task, 1])
    MSE_y = np.zeros([num_task, 1])
    RMSE_y = np.zeros([num_task, 1])

    for i in range(num_task):
        MAE_y[i] = np.mean(np.abs(error[:, i]))
        MSE_y[i] = np.mean(error[:, i] ** 2)
        RMSE_y[i] = np.sqrt(MSE_y[i])

    return MAE_y, MSE_y, RMSE_y

