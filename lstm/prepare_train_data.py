
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data




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
        self.reg = nn.Linear(mid_dim, out_dim)
        # self.reg = nn.Sequential(
        #     nn.Linear(mid_dim, mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(mid_dim, out_dim),
        # )  # regression

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
    X_train = []
    y_train = []

    for i in range(0, len(chg_list) - 16):
        train_seq_2 = []
        for j in range(i, i+15):
            train_seq_2.append([float(chg_list[j]), float(vol_list[j])])

        X_train.append(train_seq_2)
        y_train.append([[float(chg_list[i+15])]])

    X_train = X_train[:1800]
    y_train = y_train[:1800]
    print(len(X_train))

    # input_size, hidden_size, output_size, batch_size
    # model = LSTM(2, 32, 1, batch_size=args.batch_size).to(device)

    # 每个值维度 3  1 隐藏层节点数量 8 层数 1
    # net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    model = RegLSTM(2, 1, 16, 2).to(device)
    # loss_function = nn.MSELoss().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    print(X_train.shape)
    print(y_train.shape)


    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=10)


    model.to(device)


    for i in range(0, 200):
        print(f"train {i}/{len(X_train)}")

        # todo ???
        model.train()

        for X_batch, y_batch in loader:
            # print('=========================================')
            # print(y_batch)

            # print(f'X_batch: {X_batch.shape}')
            y_pred = model(X_batch)

            # print(y_pred)
            # print('=========================================')

            # print(f'y_pred: {y_pred.shape}')
            # print(f'y_batch: {y_batch.shape}')
            loss = loss_function(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    torch.save(model, './m.pth')
    # torch.save(model.state_dict(), './m.pth')

    # train_ = train_seq[-1]
    # train_ = torch.tensor(train_, device=device)
    # tag_scores = model(inputs)
    # print(tag_scores)





def test(index_code):
    index_nav_df = get_index_daily_price_as_df(index_code, 'SSE')

    # 只需要最近5年的
    index_nav_df = index_nav_df[['close','chg', 'vol']].astype(float)
    index_nav_df = index_nav_df[-5*365:]

    chg_list = index_nav_df['chg'].tolist()
    vol_list = index_nav_df['vol'].tolist()

    chg_list = normalize(chg_list)
    vol_list = normalize(vol_list)

    # 构造输入数据和验证数据
    X_train = []
    y_train = []

    for i in range(0, len(chg_list) - 16):
        train_seq_2 = []
        for j in range(i, i+15):
            train_seq_2.append([float(chg_list[j]), float(vol_list[j])])

        X_train.append(train_seq_2)
        y_train.append([[float(chg_list[i+15])]])

    # todo ???
    # model = RegLSTM(2, 1, 16, 2).to(device)
    # model.load_state_dict(torch.load('./m.pth', map_location=lambda storage, loc: storage))
    # net = model.eval()
    model = torch.load('./m.pth')

    model = model.eval()

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    print(X_train.shape)
    print(y_train.shape)


    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=10)


    model.to(device)

    for X_batch, y_batch in loader:
        # print(f'X_batch: {X_batch.shape}')
        # print(X_batch)
        y_pred = model(X_batch)

        print(y_pred)
        

if __name__ == '__main__':
    # create_train_data('000905.SH')
    test('000905.SH')
