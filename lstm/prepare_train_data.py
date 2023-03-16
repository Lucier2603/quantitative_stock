
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.optim as optim




# 000905.SH 中证500
from ts.stock_data_service import get_index_daily_price_as_df


def normalize(list):
    max_v, min_v = np.max(list), np.min(list)
    # max_v = max(list)
    # min_v = min(list)

    list = (list - min_v) / (max_v - min_v)

    return list



class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        # 每个值维度 隐藏层节点数量 层数
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 每一行代表一天 y代表预测值 y之前的每一列数据代表一个维度
def create_train_data(index_code):
    index_nav_df = get_index_daily_price_as_df(index_code, 'SSE')

    # 只需要最近5年的
    index_nav_df = index_nav_df[['close','chg', 'vol']].astype(float)
    index_nav_df = index_nav_df[-5*365:]

    chg_list = index_nav_df['chg'].tolist()
    vol_list = index_nav_df['vol'].tolist()

    chg_list = normalize(chg_list)
    vol_list = normalize(vol_list)

    # 构造输入数据和验证数据
    train_seq = []
    real_seq = []

    for i in range(0, len(chg_list) - 16):
        train_seq_2 = []
        for j in range(i, i+15):
            train_seq_2.append([chg_list[j], vol_list[j]])

        train_seq.append(train_seq_2)
        real_seq.append([chg_list[i+15]])

    # todo torch.cuda.FloatTensor


    # input_size, hidden_size, output_size, batch_size
    # model = LSTM(2, 32, 1, batch_size=args.batch_size).to(device)

    # 每个值维度 3  1 隐藏层节点数量 8 层数 1
    # net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    model = RegLSTM(2, 1, 16, 2).to(device)
    # loss_function = nn.MSELoss().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.to(device)


    for i in range(0, 200):
        print(f"train {i}/{len(train_seq)}")

        model.zero_grad()

        train_ = train_seq[i]
        real_ = real_seq[i]

        train_ = torch.tensor(train_, device=device).float()
        real_ = torch.tensor(real_, device=device).view(-1).float()  # view -1 拉平到1维

        # 那个错误应该是这里的问题 这里输入是一个2维数据?
        score = model(train_)
        score = score.float()
        loss = loss_function(score, real_)

        loss.backward()
        optimizer.step()

    train_ = train_seq[-1]
    train_ = torch.tensor(train_, device=device)
    tag_scores = model(inputs)
    print(tag_scores)






if __name__ == '__main__':
    create_train_data('000905.SH')
