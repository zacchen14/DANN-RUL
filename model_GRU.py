import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, layer_num=1, dropout=0):
        """
        定义GRU的基本结构
        :param input_num: 输入层数量
        :param hidden_num: 隐藏层数量
        :param layer_num: 层数
        :param dropout: 防止过拟合的系数
        """
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_num
        self.layer_num = layer_num
        self.dropout = dropout
        self.hidden = None  # 隐藏状态

        # 若超参数batch_first=True，则GRU的input和output以(batch_size, seqLen, input_size)三维矩阵的形式给出
        self.GRU = nn.GRU(input_size=input_num, hidden_size=hidden_num,
                          num_layers=layer_num, batch_first=True, dropout=dropout)

        self.RUL_regression = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.ReLU(True),
            nn.Linear(hidden_num // 2, hidden_num // 4),
            nn.Linear(hidden_num // 4, hidden_num // 8),
            nn.Linear(hidden_num // 8, 1),
        )

    def forward(self, input_data, alpha):
        """
        :param input_data:
        :param alpha:
        :return:
        """
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        # feature = self.feature(input_data)
        feature, self.hidden = self.GRU(input_data)
        feature = feature.view(-1, self.hidden_size)
        rul = self.RUL_regression(feature)

        return rul
