import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import os

cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36]


def get_data(path, selected_cols, rows_start, rows_end):
    """
    从csv文件中获取数据，要求数据的最后一列是RUL标签。可指定部分列和连续的若干行。
    :param path:提取的csv文件的路径
    :param selected_cols:从数据表中取部分数据时选取的列。格式是列标，元素为所选取的列，例如[1, 2, 3]
    :param rows_start:从数据表中取部分数据时开始的行标
    :param rows_end:从数据表中取部分数据时结束的行标
    :return:一个tensor dataset
    """
    cols_num = len(selected_cols)  # 选择的特征数量
    raw_train_data = pd.read_csv(path, header=0, usecols=selected_cols)
    # 取数据表中的有用信息
    x_train = raw_train_data.iloc[rows_start:rows_end, 0:cols_num - 1]
    y_train = raw_train_data.iloc[rows_start:rows_end, cols_num - 1]

    # Pytorch的tensor，要求的数据格式为numpy的float数组，以进行数据归一化
    x_train, y_train = np.array(x_train).astype(float), np.array(y_train).astype(float)
    x_train, y_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0), \
                       (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
    x_train, y_train = map(torch.Tensor, (x_train, y_train))
    # 合并为TensorDataset，便于后续处理
    train_ds: TensorDataset = TensorDataset(x_train, y_train)
    return train_ds


def predict(data, transfer=True):
    _, yb_real = data[:]

    model_root = 'models'
    rul_predict_list = np.zeros(len(data))
    if transfer:
        my_net = torch.load(os.path.join(model_root, 'model_epoch_current.pth'))
    else:
        my_net = torch.load(os.path.join(model_root, 'source_only/model_epoch_current.pth'))

    my_net = my_net.eval()

    for i in range(len(data)):
        x, y = data[i]
        x = x.view(1, 1, len(cols) - 1)
        if transfer:
            rul_output, _ = my_net.forward(x, alpha=1)
        else:
            rul_output = my_net.forward(x, alpha=1)
        rul_predict_list[i] = rul_output.item()

    rul_predict_tensor = torch.Tensor(rul_predict_list)
    loss_fn1 = nn.MSELoss(reduction='mean')
    loss_epoch = loss_fn1(rul_predict_tensor.float(), yb_real.float()).float()

    return loss_epoch
