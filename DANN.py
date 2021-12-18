# Author: Muj.ino
# Update Time: 2021.05.25 14:01
# the version of Python should >= 3.6, Pytorch >= 1.7
'''
This python file is to train the DANN model
'''

import torch
from torch.utils.data import TensorDataset
from torch import optim
import numpy as np
import pandas as pd
import sys
from model import DANNModel
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
    model = DANNModel(input_num=gen_input_size, hidden_num=gen_hidden_size, layer_num=gen_layers,
                      dropout=gen_dropout_num)
    return model, optim.Adam(model.parameters(), lr=lr)


def source_loss_batch(model, loss_func_rul=None, loss_func_domain=None, xb=None, yb=None, alpha_num=1, transfer=True):
    """
    在源域上进行RUL预测，得到预测误差，同时进行目标域/源域二分类，得到分类误差
    :param model: 模型
    :param loss_func_rul: RUL预测的损失函数
    :param loss_func_domain: 源域/目标域划分的分类的损失函数
    :param xb: 源域输入数据的x向量，包含所有特征
    :param yb: 源域输入数据的y向量，包含RUL标签
    :param alpha_num: 系数，用于抑制噪声
    :return: 源域上的分类误差和RUL预测误差
    """
    if transfer:
        domain_label = torch.zeros(len(yb)).long()  # 源域的标签为0
        rul_output, domain_output = model.forward(xb, alpha=alpha_num)
        err_s_rul = loss_func_rul(rul_output, yb)
        err_s_domain = loss_func_domain(domain_output, domain_label)

        return err_s_rul, err_s_domain, len(xb)

    else:
        rul_output = model.forward(xb, alpha=alpha_num)
        err_s_rul = loss_func_rul(rul_output, yb)

        return err_s_rul


def target_loss_batch(model, loss_func_domain, xb, yb, alpha_num):
    """
    在目标域上进行目标域/源域二分类，得到分类误差
    :param model:
    :param loss_func_domain:
    :param xb:
    :param yb:
    :param alpha_num:
    :return: 目标域上的误差
    """
    domain_label = torch.ones(len(yb)).long()  # 目标域的标签为1
    rul_output, domain_output = model.forward(xb, alpha=alpha_num)
    err_s_domain = loss_func_domain(domain_output, domain_label)

    return err_s_domain, len(xb)


def model_training(n_epochs, model, loss_func_rul=None, loss_func_domain=None, opt=None, source_ds=None,
                    target_ds=None, alpha=1, transfer=True):
    """
    滑动时间窗进行训练
    :param n_epochs: 训练总轮次
    :param model: 生成模型，这里是DANN
    :param loss_func_rul: RUL预测的损失函数
    :param loss_func_domain: domain classification的损失函数
    :param opt: 优化器
    :param source_ds: 源域的数据，TensorDataset形式
    :param target_ds: 目标域的数据，TensorDataset形式
    :param alpha: 用于抑制图像数据中的噪声
    :param transfer: 如果仅需要在源域训练，网络结构为简单的ANN，则transfer设置为Fasle，默认为True
    :return: 每一轮训练后源域和目标域预测RUL组成的误差序列
    """
    length = min(len(source_ds), len(target_ds))  # 数据集的长度length为源域和目标域中的最小长度
    loss_s = list(np.zeros(n_epochs))
    loss_t = list(np.zeros(n_epochs))

    for epoch in range(n_epochs):

        # 每epoch训练中，时间窗以步长time_window_step的步长向前移动
        for i in range(0, length - win_len + 1, sli_step):
            # 在原工程中，alpha的作用是抑制图像噪声
            # p = float(i + epoch * length) / n_epoch / length
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            start_i = i
            end_i = start_i + win_len

            # 获取源域数据
            xb_source, yb_source = source_ds[start_i:end_i]
            # 若batch_first = 1，则GRU的输入应该用(batch_size, SwqLen, input_size)的形式
            xb_source = xb_source.view(1, win_len, input_size_number)
            err_s_rul, err_s_domain, _ = source_loss_batch(model, loss_func_rul, loss_func_domain,
                                                           xb_source, yb_source, alpha)

            # 获取目标域数据
            xb_target, yb_target = target_ds[start_i:end_i]
            # 若batch_first = 1，则GRU的输入应该用(batch_size, SwqLen, input_size)的形式
            xb_target = xb_target.view(1, win_len, input_size_number)
            err_t_domain, _ = target_loss_batch(model, loss_func_domain, xb_target, yb_target, alpha)

            opt.zero_grad()     # 去掉之后，训练几轮后err变nan
            # 迁移训练需要考虑3个误差
            if transfer:
                loss = err_s_rul + err_t_domain + err_s_domain
            # 否则仅需要考虑源域的RUL预测误差
            else:
                loss = err_s_rul

            loss.backward()
            opt.step()

            sys.stdout.write('\r epoch: %d / %d, [iter: %d / all %d], err_s_rul: %f, err_s_domain: %f, err_t_domain: %f'
                             % (epoch, n_epoch, i + 1, length, err_s_rul.data.cpu().numpy(),
                                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            torch.save(model, '{0}/model_epoch_current.pth'.format(model_root))

        print('\n')
        if transfer:
            loss_s[epoch] = predict(source_data)
            print('Loss of the source dataset: %f' % loss_s[epoch])
        loss_t[epoch] = predict(target_data)
        print('Loss of the target dataset: %f\n' % loss_t[epoch])

        torch.save(model, model_save_PATH)

    return loss_s, loss_t


def train_only_source(n_epochs, model, loss_func_rul, opt, source_ds, target_ds, alpha=1):
    """
    仅在源域上训练，保存模型
    :param n_epochs: 训练总轮次
    :param model: 生成模型，这里是DANN，但不作迁移，仅在源域或目标域上单独训练
    :param loss_func_rul: RUL预测的损失函数
    :param opt: 优化器
    :param source_ds: 源域的数据，TensorDataset形式
    :param target_ds: 目标域的数据，TensorDataset形式
    :param alpha: 用于抑制图像数据中的噪声
    :return 返回模型
    """
    loss_s_rul = list(np.zeros(n_epochs))
    length = min(len(source_ds), len(target_ds))    # 作为比较，数据集长度应保持一致
    for epoch in range(n_epochs):
        # 每epoch训练中，时间窗以步长time_window_step的步长向前移动
        for i in range(0, length - win_len + 1, sli_step):
            # 在原工程中，alpha的作用是抑制图像噪声
            # p = float(i + epoch * length) / n_epoch / length
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            start_i = i
            end_i = start_i + win_len

            # 获取源域数据
            xb_source, yb_source = source_ds[start_i:end_i]
            # 若batch_first = 1，则GRU的输入应该用(batch_size, SwqLen, input_size)的形式
            xb_source = xb_source.view(1, win_len, input_size_number)
            err_s_rul, _, _ = source_loss_batch(model=model, loss_func_rul=loss_func_rul, xb=xb_source, yb=yb_source,
                                                alpha_num=alpha)

            opt.zero_grad()  # 去掉之后，训练几轮后err变nan
            loss = err_s_rul

            loss.backward()
            opt.step()

            sys.stdout.write('\r epoch: %d / %d, [iter: %d / all %d], err_s_rul: %f'
                             % (epoch, n_epoch, i + 1, length, err_s_rul.data.cpu().numpy()))
            sys.stdout.flush()
            torch.save(model, '{0}/source_only/model_epoch_current.pth'.format(model_root))

        print('\n')
        loss_s_rul[epoch] = predict(source_data, transfer=False)
        print('Loss in the target dataset: %f' % loss_s_rul[epoch])

        torch.save(model, './models/only_source/DANN_%d.pth' % n_epoch)

    return loss_s_rul


if __name__ == "__main__":

    # csv文件中所选的列。去掉功率(特征0)，离散数(特征14)，含标签RUL还有35个
    cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36]

    input_size_number = len(cols) - 1  # 最后一列(cols中的36)是标签RUL，所以特征数量-1
    hidden_size_number = 16  # 生成模型GRU的隐藏层的节点数
    num_layers = 1  # 生成模型GRU的层数
    win_len = 32  # 时间窗宽度
    sli_step = 1  # 时间窗滑动步长
    model_root = 'models'  # 保存模型的根目录

    n_epoch = 250
    lr_number = 1e-3

    loss_rul = torch.nn.MSELoss()
    loss_domain = torch.nn.NLLLoss()

    """
    ===================数据选择说明===================
    轴承温度case1 ~ case5中比较理想的退化曲线. 
    训练时取source中的带有传感器和RUL标签的数据，target中仅含有传感器数据. 
    case1: 12700 ~ 13770
    case2: 16649 ~ 17581
    case3: 15800 ~ 16897
    case4: 
    case5: 
    ===================数据选择说明===================
    """

    source_path = './data/temp_case3.csv'
    target_path = './data/temp_case1.csv'
    model_save_PATH = './models/DANN_%d.pth' % n_epoch

    source_data = get_dataset(source_path, cols, 15800, 16897)
    target_data = get_dataset(target_path, cols, 12700, 13770)

    # 若有dropout，请在下方括号加入参数，默认没有加为0
    my_model, optimizer = get_model(input_size_number, hidden_size_number, num_layers, gen_dropout_num=0, lr=lr_number)
    loss_s_list, loss_t_list = model_training(n_epoch, my_model, loss_rul, loss_domain, optimizer, source_data,
                                              target_data)
    '''
    only_source_model, only_source_optimizer = get_model(input_size_number, hidden_size_number, num_layers, 
                                                        gen_dropout_num=0, lr=lr_number)
    loss_only_source = train_only_source(n_epoch, my_model, loss_rul, loss_domain, optimizer, source_data,
                                               target_data)
    '''

    # 画图
    epoch_list = [i for i in range(n_epoch)]
    plt.figure(1)
    plt.title('loss_s in training')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, loss_s_list)
    plt.plot(epoch_list, loss_t_list)
    plt.legend(['loss_s', 'loss_t'], loc='upper right')
    plt.show()
