from SM_KISS_GP_per.SMGP import train_and_eval
from data_init import data_init, setup_seed


"""
计算的主函数

定义如下常量值
"""


# 30 50
train_iter = 50
lr = 0.1
proportion = 0.6

def main():

    # 设置随机数种子
    setup_seed(7)

    # 初始化数据
    train_x, train_y, test_x, test_y = data_init(proportion)

    MAE, MSE, RMSE, _ = train_and_eval(train_x, train_y, test_x, test_y, lr, train_iter)

    return MAE, MSE, RMSE


if __name__ == "__main__":

    MAE, MSE, RMSE = main()