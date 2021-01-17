from NN2.DGP import DGP
from pred_cal import pred_cal
from data_init import data_init, setup_seed


"""
计算的主函数

定义如下常量值
"""


batch_size = 1024
train_iter = 20
lr = 0.01
num_samples = 10
proportion = 0.6


def main():

    # 设置随机数种子
    setup_seed(2)

    # 初始化数据
    yt, yt_1, u, train_x, train_y, test_x, test_y = data_init(proportion)

    # 实例化深度 GPR 对象
    deepGP = DGP(train_x, train_y, test_x, test_y)

    # 训练，这里训练是为了拟合 g 点
    deepGP.train(lr, train_iter, batch_size)

    # 测试下 g 点的拟合效果，以及返回预测的 g 点
    MAE_g, MSE_g, RMSE_g, mean_pred = deepGP.eval(batch_size, num_samples)

    # 测试实际系统输出的预测效果
    MAE_y, MSE_y, RMSE_y = pred_cal(yt, yt_1, u, mean_pred)

    return MAE_y, MSE_y, RMSE_y, MAE_g, MSE_g, RMSE_g


if __name__ == "__main__":

    MAE_y, MSE_y, RMSE_y, MAE_g, MSE_g, RMSE_g = main()