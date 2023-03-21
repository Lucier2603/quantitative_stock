import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt



# 航班预测
# see https://www.jianshu.com/p/894268d66a5d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 读取数据
with open("./files/demo9.csv", "r", encoding="utf-8") as f:
    data = f.read()
data = [row.split(',') for row in data.split("\n")]

value_x = [[int(each[1]),int(each[2])] for each in data]
value_y = [int(each[1]) for each in data]
# 数据是每一天的航班数
li_x = []
li_y = []
seq = 2
# 因为数据集较少，序列长度太长会影响结果
for i in range(len(data) - seq):
    # 输入就是[x,x+1]天的航班数，输出时x+2天的航班数
    li_x.append(value_x[i: i+seq])
    li_y.append(value_y[i+seq])

# li_x:    [ [1,2], [1,2], [1,2] ]   142行
# li_y:    [ 1, 2, 3 ]


# 分训练和测试集
train_x = (torch.tensor(li_x[:-30]).float() / 1000.).reshape(-1, seq, 2).to(device)
train_y = (torch.tensor(li_y[:-30]).float() / 1000.).reshape(-1, 1).to(device)


print(train_x.shape)    ## 112*2*1        这里2是batch 是因为一次传入2天的数据
print(train_y.shape)    ## 112*1


test_x = (torch.tensor(li_x[-30:]).float() / 1000.).reshape(-1, seq, 2).to(device)
test_y = (torch.tensor(li_y[-30:]).float() / 1000.).reshape(-1, 1).to(device)






class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # todo DataLoader返回数据时候一般第一维都是batch，pytorch的LSTM层默认输入和输出都是batch在第二维。
        # lstm要求的入参顺序为（seq_len , btach_size , input_size）  seq是输入序列的长度 在这里应该是2？


        self.lstm = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        # 输入格式是1，输出隐藏层大小是32
        # 对于小数据集num_layers不要设置大，否则会因为模型变复杂而导致效果会变差
        # num_layers顾名思义就是有几个lstm层，假如设置成2，就相当于连续经过两个lstm层
        # 原来的输入格式是：(seq, batch, shape)
        # 设置batch_first=True以后，输入格式就可以改为：(batch, seq, shape)，更符合平常使用的习惯

        # todo 为什么是32*seq
        self.linear = nn.Linear(32*seq, 1)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x.reshape(-1, 32*seq)
        x = self.linear(x)
        return x





model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fun = nn.MSELoss()




# train

model.train()
for epoch in range(300):
    output = model(train_x)
    loss = loss_fun(output, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0 and epoch > 0:
        test_loss = loss_fun(model(test_x), test_y)
        print("epoch:{}, loss:{}, test_loss: {}".format(epoch, loss, test_loss))





# test


model.eval()
result = li_x[0][:seq-1] + list((model(train_x).data.reshape(-1))*1000) + list((model(test_x).data.reshape(-1))*1000)
# 通过模型计算预测结果并解码后保存到列表里，因为预测是从第seq个开始的，所有前面要加seq-1条数据
plt.plot(value_x, label="real")
# 原来的走势
plt.plot(result, label="pred")
# 模型预测的走势
plt.legend(loc='best')