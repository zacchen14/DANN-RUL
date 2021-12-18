import torch.nn as nn
from functions import ReverseLayerF


class DANNModel(nn.Module):

    def __init__(self, input_num, hidden_num, layer_num=1, dropout=0):
        super(DANNModel, self).__init__()  # 这一句不能去掉，不然在class内无法调用nn.Module中的内容

        self.hidden_size = hidden_num
        self.layer_num = layer_num
        self.dropout = dropout
        self.hidden = None  # 隐藏状态

        # 若超参数batch_first=True，则生成模型GRU的input和output以(batch_size, seqLen, input_size)三维矩阵的形式给出,
        # 否则(seqLen, batch_size, input_size)
        self.generator = nn.GRU(input_size=input_num, hidden_size=hidden_num,
                                num_layers=layer_num, batch_first=True, dropout=dropout)

        self.RUL_regression = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.ReLU(True),
            nn.Linear(hidden_num // 2, hidden_num // 4),
            nn.Linear(hidden_num // 4, hidden_num // 8),
            nn.Linear(hidden_num // 8, 1),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.ReLU(True),
            nn.Linear(hidden_num // 2, hidden_num // 4),
            nn.ReLU(True),
            # nn.Linear(hidden_num // 4, 2),
            # nn.Linear(hidden_num, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        # feature = self.feature(input_data)
        feature, self.hidden = self.generator(input_data)
        feature = feature.view(-1, self.hidden_size)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        rul = self.RUL_regression(feature)
        domain = self.domain_classifier(reverse_feature)

        return rul, domain
