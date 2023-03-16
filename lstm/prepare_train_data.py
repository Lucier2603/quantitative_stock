
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn



# 000905.SH 中证500
from ts.stock_data_service import get_index_daily_price_as_df


def normalize(list):
    max_v, min_v = np.max(list), np.min(list)
    # max_v = max(list)
    # min_v = min(list)

    list = (list - min_v) / (max_v - min_v)

    return list



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq[0], input_seq[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        # todo 所以是先用lstm算一遍 再linear一下，得到pred值?
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred







# 每一行代表一天 y代表预测值 y之前的每一列数据代表一个维度
def create_train_data(index_code):
    index_nav_df = get_index_daily_price_as_df(index_code, 'SSE')

    # 只需要最近5年的
    index_nav_df = index_nav_df[['close','chg', 'vol']]
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
    model = LSTM(2, 32, 1)
    # loss_function = nn.MSELoss().to(device)
    loss_function = nn.MSELoss()


    for i in range(0, len(train_seq)):
        print(f"{i}/{len(train_seq)}")

        model.zero_grad()

        train_ = train_seq[i]
        real_ = real_seq[i]

        train_ = torch.FloatTensor(train_)
        real_ = torch.FloatTensor(real_).view(-1)  # view -1 拉平到1维

        score = model(train_)
        loss = loss_function(score, real_)

        loss.backward()
        optimizer.step()


    print(1)






if __name__ == '__main__':
    create_train_data('000905.SH')
