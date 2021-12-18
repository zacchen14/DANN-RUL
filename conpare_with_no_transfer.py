# Author: Muj.ino
# Update Time: 2021.05.25 14:01
# the version of Python should >= 3.6, Pytorch >= 1.7

import torch
from torch.utils.data import TensorDataset
from torch import optim
import numpy as np
import pandas as pd
import sys
from model_GRU import GRUModel
import matplotlib.pyplot as plt
from predict import predict


def get_dataset(path, selected_cols, rows_start, rows_end):
    """
    从csv文件中获取数据，要求数据的最后一列是RUL标签。可指定部分列和连续的若干行。
    :param path: 提取的csv文件的路径
    :param selected_cols: 从数据表中取部分数据时选取的列。格式是列标，元素为所选取的列，例如[1, 2, 3]
    :param rows_start: 从数据表中取部分数据时开始的行标
    :param rows_end: 从数据表中取部分数据时结束的行标
    :return: TensorDataset，每个元素是一个元素，例如([x1, x2,..., xm], y)，其中m为特征数量
    """
    cols_num = len(selected_cols)  # 选择的特征数量
    raw_train_data = pd.read_csv(path, header=0, usecols=selected_cols)

    # 根据超参数取数据表中的有用信息
    x_train = raw_train_data.iloc[rows_start:rows_end, 0:cols_num - 1]
    y_train = raw_train_data.iloc[rows_start:rows_end, cols_num - 1]

    # 原Pytorch的tensor转变为numpy的float数组，以进行数据均一化
    x_train = np.array(x_train).astype(float)
    y_train = np.array(y_train).astype(float)
    # 数据均一化
    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)
    # 转变回tensor
    x_train, y_train = map(torch.Tensor, (x_train, y_train))

    # 合并为TensorDataset，便于后续处理
    train_ds: TensorDataset = TensorDataset(x_train, y_train)

    return train_ds


def get_model(gen_input_size, gen_hidden_size, gen_layers, gen_dropout_num, lr):
    """
    get_model: 取得DANN模型，确定模型的结构，包括输入层、隐藏层和层数。
    :param gen_input_size: DANN生成模型的输入层的特征数
    :param gen_hidden_size: DANN生成模型的输入层的隐藏层
    :param gen_layers: DANN生成模型的层数
    :param gen_dropout_num: DANN生成模型防止过拟合的系数
    :param lr: 学习率
    :return: DANN模型和优化器
    """
    model = GRUModel(input_num=gen_input_size, hidden_num=gen_hidden_size, layer_num=gen_layers,
                     dropout=gen_dropout_num)
    return model, optim.Adam(model.parameters(), lr=lr)


def source_loss_batch(model, loss_func_rul=None, xb=None, yb=None, alpha_num=1):
    """
    在源域上进行RUL预测，得到预测误差，同时进行目标域/源域二分类，得到分类误差
    :param model: 模型
    :param loss_func_rul:
    :param xb: 源域输入数据的x向量，包含所有特征
    :param yb: 源域输入数据的y向量，包含RUL标签
    :param alpha_num: 系数，用于抑制噪声
    :return: 源域上的分类误差和RUL预测误差
    """
    rul_output = model.forward(xb, alpha=alpha_num)
    err_s_rul = loss_func_rul(rul_output, yb)

    return err_s_rul


def model_training(n_epochs, model, loss_func_rul=None, opt=None, source_ds=None, target_ds=None, alpha=1):
    """
    滑动时间窗进行训练
    :param n_epochs: 训练总轮次
    :param model: 生成模型，这里是GRU
    :param loss_func_rul: RUL预测的损失函数
    :param opt: 优化器
    :param source_ds: 源域的数据，TensorDataset形式
    :param target_ds: 目标域的数据，TensorDataset形式
    :param alpha: 用于抑制图像数据中的噪声
    :return: 每一轮训练后源域和目标域预测RUL组成的误差序列
    """
    length = min(len(source_ds), len(target_ds))  # 数据集的长度length为源域和目标域中的最小长度
    loss_t = list(np.zeros(n_epochs))

    for epoch in range(n_epochs):

        # 每epoch训练中，时间窗以步长time_window_step的步长向前移动
        for i in range(0, length - time_window_length + 1, time_window_step):
            # 在原工程中，alpha的作用是抑制图像噪声
            # p = float(i + epoch * length) / n_epoch / length
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            start_i = i
            end_i = start_i + time_window_length

            # 获取源域数据
            xb_source, yb_source = source_ds[start_i:end_i]
            # 若batch_first = 1，则GRU的输入应该用(batch_size, SwqLen, input_size)的形式
            xb_source = xb_source.view(1, time_window_length, input_size_number)
            err_s_rul = source_loss_batch(model, loss_func_rul, xb_source, yb_source, alpha)

            opt.zero_grad()  # 去掉之后，训练几轮后err变nan

            loss = err_s_rul

            loss.backward()
            opt.step()

            sys.stdout.write('\r epoch: %d / %d, [iter: %d / all %d], err_s_rul: %f'
                             % (epoch, n_epoch, i + 1, length, err_s_rul.data.cpu().numpy()))
            sys.stdout.flush()
            torch.save(model, '{0}/model_epoch_current.pth'.format(model_root))

        print('\n')

        loss_t[epoch] = predict(target_data)
        print('Loss of the target dataset: %f\n' % loss_t[epoch])

    torch.save(model, model_save_PATH)

    return loss_t


if __name__ == "__main__":
    # csv文件中所选的列。去掉功率(特征0)，离散数(特征14)，含标签RUL还有35个
    cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36]

    input_size_number = len(cols) - 1  # 最后一列(cols中的36)是标签RUL，所以特征数量-1
    hidden_size_number = 16  # 生成模型GRU的隐藏层的节点数
    num_layers = 1  # 生成模型GRU的层数
    time_window_length = 32  # 时间窗宽度
    time_window_step = 1  # 时间窗滑动步长
    model_root = 'models'  # 保存模型的根目录

    loss_rul = torch.nn.MSELoss()

    n_epoch = 200
    lr_number = 1e-3

    source_path = './data/temp_case3.csv'
    target_path = './data/temp_case1.csv'
    model_save_PATH = './models/compare/GRU_%d.pth' % n_epoch

    source_data = get_dataset(source_path, cols, 15800, 16897)
    target_data = get_dataset(target_path, cols, 12700, 13770)

    epoch_list = [i for i in range(n_epoch)]
    loss_s_list = list(np.zeros(n_epoch))

    # 若有dropout，请在下方括号加入参数，默认没有加为0
    my_model, optimizer = get_model(input_size_number, hidden_size_number, num_layers, gen_dropout_num=0, lr=lr_number)
    loss_s_list = model_training(n_epoch, my_model, loss_rul, opt=optimizer, target_ds=target_data,
                                 source_ds=source_data)

    # 画图
    plt.figure(1)
    plt.title('loss_s in training')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, loss_s_list)
    plt.legend(['loss_s'], loc='upper right')
    plt.show()
