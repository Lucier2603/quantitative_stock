
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
    list = (list - min_v) / (max_v - min_v)

    return [float(x) for x in list]


seq = 15

class RegLSTM(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(RegLSTM, self).__init__()

        # batch_first=True的意义
        # DataLoader返回数据时候一般第一维都是batch，pytorch的LSTM层默认输入和输出都是batch在第二维。
        # lstm要求的入参顺序为（seq_len , btach_size , input_size）  seq是输入序列的长度 在这里应该是15  batch_size应该是总样本数量

        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=32, num_layers=1, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32
        # 对于小数据集num_layers不要设置大，否则会因为模型变复杂而导致效果会变差
        # num_layers顾名思义就是有几个lstm层，假如设置成2，就相当于连续经过两个lstm层
        # 原来的输入格式是：(seq, batch, shape)
        # 设置batch_first=True以后，输入格式就可以改为：(batch, seq, shape)，更符合平常使用的习惯
        # todo 为什么是32*seq
        # Linear 参数  mid_dim, out_dim
        self.linear = nn.Linear(32 * seq, y_dim)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 32*seq)
        x = self.linear(x)
        return x






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 每一行代表一天 y代表预测值 y之前的每一列数据代表一个维度
def create_train_data(index_code):
    # step 1.1 获取最近8年的数据
    index_nav_df = get_index_daily_price_as_df(index_code, 'SSE')
    index_nav_df = index_nav_df[['close','chg', 'vol']].astype(float)
    index_nav_df = index_nav_df[-8*365:]

    # step 2.1 计算出x指标数据
    x_raw_data = {}
    x_raw_names = ['chg', 'vol']
    x_raw_data['chg'] = index_nav_df['chg'].tolist()
    x_raw_data['vol'] = index_nav_df['vol'].tolist()

    # step 2.2 计算出y指标数据
    y_raw_names = ['chg']
    y_raw_data = index_nav_df['chg'].tolist()

    x_dim = len(x_raw_names)
    y_dim = len(y_raw_names)

    # step 2.2 数据标准化
    for x_name in x_raw_names:
        l = x_raw_data[x_name]
        l = normalize(l)
        x_raw_data[x_name] = l

    y_raw_data = [float(x) for x in y_raw_data]


    # step 3.1 构造输入数据和验证数据
    print('step 3.1 build data')
    x_list = []
    y_list = []
    seq = 15

    for i in range(0, len(index_nav_df) - seq):
        xp = []

        # 一个样本有seq条数据点
        for j in range(i, i + seq):
            # 一条数据点有n个维度的数据
            x = []
            for name in x_raw_names:
                d = x_raw_data[name][j]
                x.append(d)

            xp.append(x)
        x_list.append(xp)

    y_list = y_raw_data

    # step 3.2 转换到tensor数据
    x_list = x_list[:2700]
    y_list = y_list[:2700]

    x_train = x_list[:2100]
    y_train = y_list[:2100]
    x_test = x_list[2100:]
    y_test = y_list[2100:len(x_list)]

    X_train = torch.tensor(x_train).reshape(-1, seq, x_dim).to(device)
    y_train = torch.tensor(y_train).reshape(-1, y_dim).to(device)
    X_test = torch.tensor(x_test).reshape(-1, seq, x_dim).to(device)
    y_test = torch.tensor(y_test).reshape(-1, y_dim).to(device)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    # step 4.1 训练
    print('step 4.1 start train')
    model = RegLSTM(x_dim, y_dim).to(device)
    loss_fun = nn.MSELoss().to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


    model.train()
    for epoch in range(300):
        output = model(X_train)
        loss = loss_fun(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 and epoch > 0:
            test_loss = loss_fun(model(X_test), y_test)
            print("epoch:{}, loss:{}, test_loss: {}".format(epoch, loss, test_loss))

    torch.save(model, './m.pth')




if __name__ == '__main__':
    create_train_data('000905.SH')
    # test('000905.SH')
