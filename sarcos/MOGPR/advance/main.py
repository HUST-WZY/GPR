from data_init import setup_seed, data_init
from pred_cal import pred_cal
from MOGPR.GPR import train_and_eval


"""
计算的主函数

定义如下常量值
"""


# 30 50
train_iter = 50
lr = 0.01
proportion = 0.6
batch_size = 1024 * 4

def main():

    # 设置随机数种子
    setup_seed(7)

    # 初始化数据
    yt, yt_1, u, train_x, train_y, test_x, test_y = data_init(proportion)

    # 训练，这里训练是为了拟合 g 点
    MAE_g, MSE_g, RMSE_g, mean_pred = train_and_eval(train_x, train_y, test_x, test_y, batch_size, lr, train_iter)

    # 测试实际系统输出的预测效果
    MAE_y, MSE_y, RMSE_y = pred_cal(yt, yt_1, u, mean_pred)

    return MAE_y, MSE_y, RMSE_y, MAE_g, MSE_g, RMSE_g


if __name__ == "__main__":

    MAE_y, MSE_y, RMSE_y, MAE_g, MSE_g, RMSE_g = main()