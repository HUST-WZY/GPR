# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 22:58
# @Author  : wzy
# @FileName: data_loader.py
import random
import torch

def loader(data_x, data_y, batch):
    random.seed(1)

    if len(data_x.shape) == 1:
        num_input = 1
    else:
        num_input = data_x.shape[1]

    if len(data_y.shape) == 1:
        num_tasks = 1
    else:
        num_tasks = data_y.shape[1]

    data = torch.cat([data_x.reshape(-1, num_input), data_y.reshape(-1, num_tasks)], dim=1)
    n = data.shape[0]
    index = list(range(n))
    random.shuffle(index)

    ans_x = []
    ans_y = []

    i = -1

    for i in range(0, n, batch):

        if (i + batch) <= n:

            ans_x.append(data[index[i : i + batch], : num_input])
            ans_y.append(data[index[i : i + batch], num_input :])

    if (i + batch) != n:

        ans_x.append(data[index[n - batch : n], : num_input])
        ans_y.append(data[index[n - batch : n], num_input :])

    return ans_x, ans_y


def main():

    data_x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    data_y = torch.tensor([[1], [3], [5], [7], [9], [11], [13], [15]])


    ans_x, ans_y = loader(data_x, data_y, 3)

    for i in range(len(ans_x)):
        print(ans_x[i])
        print(ans_y[i])
        print("--------")




if __name__ == "__main__":
    main()