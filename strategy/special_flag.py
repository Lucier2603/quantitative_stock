

import pandas as pd
import datetime

from strategy.assist import ma_i

'''
10日线偏离度
'''
def deviation_degree_on_ma10(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df

    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()

    # 振幅
    stock_df['amp'] = stock_df['high']-stock_df['low']
    stock_df['amp'] = stock_df['amp']/stock_df['close'].shift(1)-1
    stock_df['amp_avg'] = stock_df['amp'].rolling(20).mean()

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0


    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        buy_flag_1 = True

        # 买入条件 1.
        if r['low'] / ma_i(stock_df, 'close', 10, i) > (1-r['amp']*12):
            continue

        if buy_flag_1 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'close':r['open'], 'price':r['open'], 'type':'B'}, ignore_index=True)


        # sell
        # 下穿5日线 卖出
        ma_10 = ma_i(stock_df, 'close', 5, i)
        sell_flag_1 = r['close'] <ma_10
        # 触达止损点 卖出   止损设置在-5%
        sell_flag_2 = r['low'] < last_buy_price * 0.95

        if last_bs_type == 'B' and last_buy_price != r['close']:
            if sell_flag_1 or sell_flag_2:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'close': r['close'], 'price': r['close'], 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'close': r['close'], 'price': r['close'],
             'type': 'N'}, ignore_index=True)

    return bs_df