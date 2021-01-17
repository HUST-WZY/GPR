from NN2.DGP import DGP
from data_init import data_init, setup_seed


"""
计算的主函数

定义如下常量值
"""


# 30 50
train_iter = 20
lr = 0.01
proportion = 0.6
batch_size = 1024

def main():

    # 设置随机数种子
    setup_seed(7)

    # 初始化数据
    train_x, train_y, test_x, test_y = data_init(proportion)

    dgp = DGP(train_x, train_y, test_x, test_y)

    # 训练，这里训练是为了拟合 g 点
    dgp.train(lr, train_iter, batch_size)

    # 测试下 g 点的拟合效果，以及返回预测的 g 点
    MAE, MSE, RMSE, _ = dgp.eval(batch_size, num_samples=10)

    return MAE, MSE, RMSE


if __name__ == "__main__":

    MAE, MSE, RMSE = main()