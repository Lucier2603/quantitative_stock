
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,  # 输入尺寸为 1，表示一天的数据
            hidden_size=128,
            num_layers=1,
            batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(128, 1))

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全 0 的 state
        out = self.out(r_out[:, -7:, :])  # 取最后一天作为输出

        return out


