
import pandas as pd
import numpy as np



# 000905.SH 中证500
from ts.stock_data_service import get_index_daily_price_as_df

# 每一行代表一天 y代表预测值 y之前的每一列数据代表一个维度
def create_train_data(index_code):
    index_nav_df = get_index_daily_price_as_df(index_code, 'SSE')

    # 创建训练集
    days_before = 10

    series = index_nav_df['close'].copy()

    train_series, test_series = series[:500], series[490:]

    train_data = pd.DataFrame()

    # 通过移位，创建历史 days_before 天的数据
    for i in range(10):
        # 当前数据的 7 天前的数据，应该取 开始到 7 天前的数据； 昨天的数据，应该为开始到昨天的数据，如：
        # [..., 1,2,3,4,5,6,7] 昨天的为 [..., 1,2,3,4,5,6]
        # 比如从 [2:-7+2]，其长度为 len - 7
        train_data['c%d' % i] = train_series.tolist()[i: -days_before + i]

    # 获取对应的 label
    train_data['y'] = train_series.tolist()[days_before:]

    # 是否生成 index
    print(1)





    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)

    # 行转换成列
    seq_number = seq_number[:, np.newaxis]

    # print(repr(seq))
    # 1949~1960, 12 years, 12*12==144 month
    # 结果 seq_year_month 有两列，列0代表year从0开始，列1代表month从0开始   like (0,0) (0,1) ... (0,11) (1,0) ...
    seq_year = np.arange(12)
    seq_month = np.arange(12)
    seq_year_month = np.transpose(
        [np.repeat(seq_year, len(seq_month)),
         np.tile(seq_month, len(seq_year))],
    )  # Cartesian Product

    # 简单的merge
    seq = np.concatenate((seq_number, seq_year_month), axis=1)

    # 如上的作用是形成  price, year, month 的三元组合

    # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)

    print(1)






if __name__ == '__main__':
    create_train_data('000905.SH')
