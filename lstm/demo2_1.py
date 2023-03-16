import pandas as pd
import numpy as np
import torch



def load_data():
    df = pd.read_csv('D:\\1.csv')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    return df



def nn_seq_us(B):
    print('data processing...')
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        load = load.tolist()
        data = data.values.tolist()
        load = (load - n) / (m - n)
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            # 每一个时间点有24天的数据，每一天的数据只有一个维度，记在x里
            for j in range(i, i + 24):
                # x代表一天的数据，只有1个维度
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])

            # todo 这个应该是第25天的结果
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)

            # todo 于是，train代表输入，label代表实际结果
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)

    return Dtr, Val, Dte, m, n






class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
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


def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    # todo 这个loss函数后面怎么使用
    loss_function = nn.MSELoss().to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)


    # StepLR 学习率
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5

    # tqdm 进度条
    # todo args.epochs ?
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)

            y_pred = model(seq)
            loss = loss_function(y_pred, label)

            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # todo 这个step什么意思？上一个循环的Dtr，永远都是不变的？
        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

        # todo ?
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

if __name__ == '__main__':
    nn_seq_us(3)
