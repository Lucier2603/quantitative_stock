import json

import multiprocessing as mp

import numpy

from strategy.assist import ma, bs_test_a, ma_i, high_vs_close, cal_atr
from ts.stock_data_service import get_all_stock_code, get_stock_daily_price_as_df, get_all_stock

import datetime
import pandas as pd




# 买入点
bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])



'''
上穿30日线 策略
'''
def strategy_more_than_ma_30(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df

    if stock_df.iloc[-1]['high'] < 5:
        return bs_df

    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()

    ma(stock_df, 'close', 10)
    ma(stock_df, 'close', 20)
    ma(stock_df, 'close', 30)

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0


    low_30_days = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



    for i in stock_df.index:
        r = stock_df.loc[i]
        cur_date = r['trade_date']

        buy_flag_1 = True
        buy_flag_2 = False
        buy_flag_3 = False

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        # 买入条件 1. 前五日 至少三日 收盘低于30日线
        # less_than_30_days = 0
        # for j in range(1,6):
        #     if stock_df.loc[i-j]['close']>r['close_ma30']:
        #         less_than_30_days+=1
        # if less_than_30_days<3:
        #     buy_flag_1 = False

        # 买入条件 2. 当日必须收涨
        # if r['close']<r['open']:
        #     buy_flag_2 = False

        # 买入条件 3. 5日内价格必须在最高的10%以上
        highest_5 = 0
        for j in range(1, 6):
            rr = stock_df.loc[i-j]
            if rr['close']>highest_5:
                highest_5 = rr['close']
        if r['close'] < highest_5*0.95:
            continue


        # 买入条件 4. 20日线上穿30日线 买入
        # if r['close_ma10'] > r['close_ma20'] and r['close_ma20'] > r['close_ma30']:
        #     buy_flag_4 = True
        # else:
        #     buy_flag_4 = False

        # 买入条件 5. 上穿20日线
        r_1 = stock_df.loc[i-1]
        if r['high']>r['close_ma20'] and r['low']<r['close_ma20'] and r_1['high']<r['close_ma20']:
            buy_flag_5 = True
        else:
            buy_flag_5 = False

        # 并且30日线处于上升状态
        # buy_flag_1 = buy_flag_1 and stock_df.loc[i-5, 'close_ma30'] < r['close_ma30']

        # todo vol涨多跌少


        if buy_flag_1 and buy_flag_2 and buy_flag_3 and (last_bs_type != 'B'):
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



'''
寻找趋势 策略
'''
def strategy_find_trend_A(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    # if stock_code != '000001.SZ':
    #     return bs_df

    if len(stock_df) == 0:
        return bs_df

    if stock_df.iloc[-1]['high'] < 5:
        return bs_df

    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()

    #
    N = 20
    ma(stock_df, 'close', 10)
    ma(stock_df, 'close', 20)
    ma(stock_df, 'close', 30)
    cal_atr(stock_df, 12)
    # 跌幅大于-5的标记
    stock_df['flag_fail_than_5'] = stock_df['chg'].apply(lambda x: 1 if x<-5 else 0)
    stock_df['flag_fail_than_5_sum'] = stock_df['flag_fail_than_5'].rolling(N).sum()
    # 20日内振幅10%以上days


    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0


    for i in stock_df.index:
        r = stock_df.loc[i]
        cur_date = r['trade_date']

        buy_flag_1 = True
        buy_flag_2 = True
        buy_flag_3 = True
        buy_flag_4 = True
        buy_flag_5 = True

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        # 买入条件 1. 连续10日高于 ma30 & ma20
        for j in range(0, 10):
            r2 = stock_df.loc[i-j]
            if (r2['close'] < r2['close_ma30']) or (r2['close'] < r2['close_ma20']):
                buy_flag_1 = False
                continue

        # 买入条件 2.杜绝短期高潮涨势 (波动率)
        if r['ATR'] > 10:
            continue

        # 买入条件 3.20day 无超过2次的5%以上暴跌
        if r['flag_fail_than_5_sum'] >= 2:
            continue

        # 买入条件 4. 最近8日阳阴平均成交量大于0.8
        up_total_vol, up_days = 0, 0
        down_total_vol, down_days = 0, 0
        for j in range(0, 8):
            r2 = stock_df.loc[i - j]
            if r2['chg'] >= 0:
                up_total_vol += r2['vol']
                up_days += 1
            else:
                down_total_vol += r2['vol']
                down_days += 1
        if up_days*down_days != 0:
            buy_flag_4 = (down_total_vol/down_days) / (up_total_vol/up_days) < 0.8

        # 买入条件 5. 10日线偏离度


        # 买入条件 6. 20日内振幅10%以上占比不超过3天
        cnt_10 = 0
        for j in range(0, 20):
            r2 = stock_df.loc[i-j]
            r3 = stock_df.loc[i-j-1]
            if abs(r2['high']-r2['low'])/r3['close'] > 0.12:
                cnt_10 += 1
        if cnt_10 > 3:
            continue



        if buy_flag_1 and buy_flag_2 and buy_flag_3 and buy_flag_4 and buy_flag_5 and (last_bs_type != 'B'):
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



'''
寻找趋势 策略
'''
def strategy_find_trend_B(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df

    if stock_df.iloc[-1]['high'] < 5:
        return bs_df

    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()


    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0


    for i in stock_df.index:
        r = stock_df.loc[i]
        cur_date = r['trade_date']

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        # 买入条件 1.
        for j in range(0, 20):
            r2 = stock_df.loc[i-j]
            if r2['high'] < ma_i(stock_df, 'close', 10, i-j):
                continue

        if (last_bs_type != 'B'):
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


'''
行业共振 策略
'''
def strategy_industry_top(stock_code, stock_df):
    return 1








# todo z score 选股策略







'''
大阳后小幅回调 策略

关注的点：
1. 是否近期有前高爆炒
2. 下行均线压制明显
3. 第一根极高交易量的大阳线

5. 均线位置和方向
6. 阳线需要短上影线 并且第二天的回调不能是高开下砸
7. 688的票少碰

'''
def strategy_after_big_increase(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 10:
        return bs_df

    # 限定只看8元以上的
    if stock_df.loc[len(stock_df)-1]['close'] < 8:
        return bs_df

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 30)

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0

    for i in stock_df.index:
        r = stock_df.loc[i]
        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or r['trade_date'] > end_strategy_time:
            continue
        if i < 60:
            continue

        # 计算辅助指标
        # ma_vol_30 = ma_i(stock_df, 'vol', 30, i)



        # buy
        buy_flag_1 = True
        buy_flag_2 = False

        # 前3天内 2天合计涨幅达9%
        if stock_df.loc[i-3]['chg']+stock_df.loc[i-2]['chg']<9:
            buy_flag_1 = False
        big_rise_close = stock_df.loc[i-2]['close']
        big_rise_vol = stock_df.loc[i-2]['vol'] if stock_df.loc[i-2]['chg']>stock_df.loc[i-3]['chg'] else stock_df.loc[i-3]['vol']

        # 5天内涨幅不能过大(剔除连续涨停)
        chg_10_day = (stock_df.loc[i-2]['close']-stock_df.loc[i-7]['close'])/stock_df.loc[i-12]['close']-1
        if chg_10_day>0.25:
            buy_flag_1 = False

        # 判断回调 需要缩量盘整
        if stock_df.loc[i-1]['chg'] < 1 and stock_df.loc[i-1]['chg'] > -3:
            if stock_df.loc[i]['chg'] < 1 and stock_df.loc[i]['chg'] > -3:
                if (stock_df.loc[i-1]['vol'] < big_rise_vol*0.6) or (stock_df.loc[i]['vol'] < big_rise_vol*0.6):
                    buy_flag_2 = True
        # 若有跌幅大于-1%的回调 则其必须缩量
        if stock_df.loc[i-1]['chg'] < -1 and stock_df.loc[i-1]['vol'] > big_rise_vol*0.7:
            buy_flag_1 = False
        if stock_df.loc[i-2]['chg'] < -1 and stock_df.loc[i-2]['vol'] > big_rise_vol*0.7:
            buy_flag_1 = False


        # 回调日不能高开下砸
        # if stock_df.loc[i-1]['open'] > stock_df.loc[i-2]['high']:
        #     buy_flag_1 = False

        # 当日涨幅不能超过2% 不能低于-3%
        if stock_df.loc[i]['chg']>2 or stock_df.loc[i]['chg']<-3:
            buy_flag_1 = False

        # 当日不能高开下砸
        if (stock_df.loc[i]['high']-stock_df.loc[i]['low'])/stock_df.loc[i]['low']>6:
            buy_flag_1 = False


        if buy_flag_1 and buy_flag_2 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'price':r['close'], 'type':'B'}, ignore_index=True)


        # sell

        # 触达止损点 卖出   止损设置在-7%
        sell_flag_1 = r['low'] < last_buy_price * 0.93
        if sell_flag_1:
            sell_price = r['low']
        # 收益达到 10% 卖出
        sell_flag_2 = (last_buy_price!=0) and r['close'] > last_buy_price * 1.07
        if sell_flag_2:
            sell_price = last_buy_price * 1.07

        if last_bs_type == 'B':
            if sell_flag_1 or sell_flag_2:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'}, ignore_index=True)

    return bs_df




# todo 卖出策略 持有期间最高价下跌10%



'''
阳线量大 阴线量小 策略

'''
def strategy_rise_high_vol_vs_down_low_vol(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 10:
        return bs_df

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 30)

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or r['trade_date'] > end_strategy_time:
            continue
        if i < 60:
            continue

        # 计算辅助指标
        # ma_vol_30 = ma_i(stock_df, 'vol', 30, i)

        # buy
        buy_flag_1 = True
        buy_flag_2 = True
        buy_flag_3 = True

        # 条件1 当天必须是阴线
        if r['open'] < r['close']:
            buy_flag_1 = False

        # 条件2 5天内的vol比较
        rise_total_vol = 0
        rise_days = 0
        down_total_vol = 0
        down_days = 0
        # 最大的交易量 大于0表示上涨 小于0表示下跌
        max_vol = 0
        for j in range(0,5):
            rt = stock_df.loc[i-j]
            # 必须收盘收在10日线上
            # if rt['close']<rt['close_ma10']:
            #     buy_flag_2 = False
            #     continue

            if rt['close']>rt['open']:
                # 上涨
                rise_total_vol += rt['vol']
                rise_days+=1
                max_vol = rt['vol'] if rt['vol']>abs(max_vol) else max_vol
            else:
                # 下跌
                down_total_vol += rt['vol']
                down_days += 1
                max_vol = -rt['vol'] if rt['vol'] > abs(max_vol) else max_vol

                # 不能有大跌
                if rt['chg']<-4:
                    buy_flag_2 = False

        if down_days > 1 or down_days == 0:
            buy_flag_2 = False
        else:
            buy_flag_2 = True

        if rise_days*down_days!=0 and (down_total_vol / down_days) / (rise_total_vol / rise_days) > 0.7:
            buy_flag_3 = False

        # 最大交易量那天 必须是上涨
        if max_vol<0:
            buy_flag_3 = False



        if buy_flag_1 and buy_flag_2 and buy_flag_3 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'price':r['close'], 'type':'B'}, ignore_index=True)


        # sell

        # 触达止损点 卖出   止损设置在-7%
        sell_flag_1 = r['low'] < last_buy_price * 0.93
        if sell_flag_1:
            sell_price = r['low']
        # 收益达到 5% 卖出
        sell_flag_2 = (last_buy_price!=0) and r['close'] > last_buy_price * 1.05
        if sell_flag_2:
            sell_price = last_buy_price * 1.07

        if last_bs_type == 'B':
            if sell_flag_1 or sell_flag_2:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'}, ignore_index=True)

    return bs_df




'''
龙头战法 策略
仅供研究使用

'''
def strategy_first_fall_in_successive_up(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 10:
        return bs_df

    ma(stock_df, 'close', 5)


    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0
    last_buy_date = 0

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or r['trade_date'] > end_strategy_time:
            continue
        if i < 60:
            continue

        # buy
        buy_flag_1 = True

        # if r['close'] > 5:
        #     buy_flag_1 = False

        # 除了最后一天，前4天连续上涨10%
        for j in range(2,6):
            rt = stock_df.loc[i-j]
            if rt['chg']<9:
                buy_flag_1 = False

        # 最后前一天必须要收阴线
        if stock_df.loc[i-1]['close'] > stock_df.loc[i-1]['open']:
            buy_flag_1 = False

        # 当天开盘必须-6起步
        if (r['open']-stock_df.loc[i-1]['close'])/stock_df.loc[i-1]['close'] > -0.06:
            buy_flag_1 = False
        # todo 当天没有长上影线
        if (r['high']-r['close'])/r['close']>0.08:
            buy_flag_1 = False

        # 当天最低价必须到过-9.5
        if (r['low']-stock_df.loc[i-1]['close'])/stock_df.loc[i-1]['close'] > -0.095:
            buy_flag_1 = False
        # 当天开盘必须在5日线之下
        if (r['open']-r['close_ma5'])/r['close_ma5']>-0.08:
            buy_flag_1 = False

        if buy_flag_1 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['open']
            last_buy_date = i
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'price':stock_df.loc[i-1]['close']*0.905, 'type':'B'}, ignore_index=True)


        # sell
        # 触达止损点 卖出   止损设置在-7%
        # sell_flag_1 = r['low'] < last_buy_price * 0.93
        # if sell_flag_1:
        #     sell_price = r['low']
        # 收益达到 10% 卖出
        # sell_flag_2 = (last_buy_price!=0) and r['close'] > last_buy_price * 1.05
        # if sell_flag_2:
        #     sell_price = last_buy_price * 1.07

        # 机械性的在第二天卖出
        sell_flag_1 = False
        if i == last_buy_date + 1:
            sell_flag_1 = True
            # 确定卖出价格
            if r['high'] > last_buy_price * 1.05:
                sell_price = last_buy_price * 1.05
            else:
                sell_price = r['close']

        if last_bs_type == 'B':
            if sell_flag_1:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'}, ignore_index=True)

    return bs_df




'''
缩量大跌 策略
仅供研究使用

'''
def strategy_fall_with_high_chg_and_low_vol(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 10:
        return bs_df

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or r['trade_date'] > end_strategy_time:
            continue
        if i < 60:
            continue

        # buy
        buy_flag_1 = True
        buy_flag_2 = True

        # 当日大跌，且缩量
        # todo 可能不仅要大跌 还要跌破均线 逼迫止损
        if r['chg']>-5 or r['chg']<-9.5:
            buy_flag_1 = False

        # 之前5天内，找到上涨的并且量最大的阳线
        max_vol = 0
        for j in range(1,6):
            rt = stock_df.loc[i-j]
            if rt['chg']>1 and rt['vol']>max_vol:
                max_vol=rt['vol']

        if r['vol']<max_vol*0.7:
            buy_flag_2 = True
        else:
            buy_flag_2 = False


        if buy_flag_1 and buy_flag_2 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'price':r['close'], 'type':'B'}, ignore_index=True)


        # sell

        # 触达止损点 卖出   止损设置在-7%
        sell_flag_1 = r['low'] < last_buy_price * 0.93
        if sell_flag_1:
            sell_price = r['low']
        # 收益达到 10% 卖出
        sell_flag_2 = (last_buy_price!=0) and r['close'] > last_buy_price * 1.05
        if sell_flag_2:
            sell_price = last_buy_price * 1.07

        if last_bs_type == 'B':
            if sell_flag_1 or sell_flag_2:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'}, ignore_index=True)

    return bs_df











# 策略 波段 第一次回调在10日 第二次在20日 第三次在30日 第四次到30日 就会破


'''
小阳堆积 策略

止损条件
1. 回调跌破突破大阳线实体的1/2 (待测试)
2. 下跌后紧跟的上涨,未能超过下跌的高点
3. 连续跌破5日/10日线
'''
def strategy_successive_small_up(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 100:
        return bs_df

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 30)

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0
    last_buy_date = 0


    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or r['trade_date'] > end_strategy_time:
            continue
        if i < 60:
            continue

        # 计算辅助指标
        ma_5 = ma_i(stock_df, 'close', 5, i)
        ma_10 = ma_i(stock_df, 'close', 10, i)

        # buy
        buy_flag_1 = True
        # 必须先有一个大于4的涨幅
        # if stock_df.loc[i-6]['chg'] < 4:
        #     buy_flag_1 = False
        # 之前不能有连续的大涨
        if stock_df.loc[i-7]['chg']>7 or stock_df.loc[i-8]['chg']>7 or stock_df.loc[i-9]['chg']>7:
            buy_flag_1 = False

        # 前连续4日满足小阳条件
        total_chg = 0
        # 下跌的天数最多一天
        down_days = 0
        # 阴线日的成交量
        green_day_vol = 0
        # 收阴线的天数
        green_days = 0
        # 阳线日的成交量
        red_day_vol = 0
        # 收阳线的天数
        red_days = 0
        # 涨幅大于3的天数
        chg_more_than_3_days = 0
        # 涨幅大于7的天数
        chg_more_than_6_days = 0

        # 最高价格
        highest = stock_df.loc[i - 6]['high']
        # 统计未超过最高价格的天数
        less_than_highest_days = 0

        for j in range(1,6):
            rt = stock_df.loc[i-j]
            total_chg += rt['chg']
            # 统计下跌天数
            if rt['chg'] > -2 and rt['chg'] < -0.5:
                down_days += 1
            # 统计收阴线天数
            if rt['close'] < rt['open']:
                green_days += 1
                green_day_vol += rt['vol']
            else:
                red_days += 1
                red_day_vol += rt['vol']
            # 不能有高开下砸大阴线 超过4个点振幅
            if rt['close']<rt['open'] and rt['open']>stock_df.loc[i-j-1]['close']:
                if (rt['high']-rt['low'])/rt['low']>4:
                    buy_flag_1 = False
                    break
            # 不能有长下影线
            if ( min(rt['close'],rt['open']) - rt['low'])/rt['low']>0.04:
                buy_flag_1 = False
                break
            # 不能有长上影线
            if (rt['high'] - max(rt['close'],rt['open']))/rt['high']>3.2:
                buy_flag_1 = False
                break

            # 涨幅必须小
            if rt['chg'] < -2:
                buy_flag_1 = False
                break
            if rt['chg'] > 3:
                chg_more_than_3_days += 1
            if rt['chg'] > 7:
                chg_more_than_6_days += 1

            # todo 这个靠看图 不能接近 均线等 关键点位


            # todo 关注低点抬升 高点抬升不太可能

            # 必须在5日线上
            ma_5_j = ma_i(stock_df, 'close', 5, i-j)
            if rt['close'] < ma_5_j:
                buy_flag_1 = False
                break

        # 至少有一天涨幅大于3
        if chg_more_than_3_days<1:
            buy_flag_1 = False
        # 最多有一天涨幅大于7
        if chg_more_than_6_days>1:
            buy_flag_1 = False
        # 最近3天每日涨幅不能大于3
        if stock_df.loc[i-1]['chg']>3 or stock_df.loc[i-2]['chg']>3 or stock_df.loc[i-3]['chg']>3:
            buy_flag_1 = False
        # 下跌天数最多一天
        if down_days > 1:
            buy_flag_1 = False
        # 收阴线天数最多一天
        if green_days > 1:
            buy_flag_1 = False
            continue
        # 阳线成交量必须大于阴线成交量
        if green_days!=0 and red_days!=0 and (green_day_vol/green_days)/(red_day_vol/red_days) > 0.7:
            buy_flag_1 = False
        # 4日内平均涨幅不大
        avg_chg = total_chg/4
        if avg_chg<0.5 or avg_chg>3:
            buy_flag_1=False
        # 当日必须下跌 并且跌幅必须小于-3 并且收在5日线的99%上 并且当日不能大涨
        if r['close']>r['open']:
            buy_flag_1 = False
        if r['chg']<-3 or r['close']<ma_5*0.98 or r['chg']>avg_chg*2 or r['chg']>3:
            buy_flag_1 = False
        # 如果当天跌幅较大 则其成交量不能显著放大
        if r['chg']<-0.5:
            if red_days!=0 and r['vol']>(0.9*red_day_vol/red_days):
                buy_flag_1 = False


        # todo 加一个奇怪的条件 必须超过10元
        if r['close'] < 10:
            continue

        if buy_flag_1 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            last_buy_date = r['trade_date']
            bs_df = bs_df.append(
                {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': r['close'], 'type': 'B'},
                ignore_index=True)

        # sell
        # 收益达到 10% 卖出
        sell_flag_3 = (last_buy_price != 0) and r['high'] > last_buy_price * 1.10
        if sell_flag_3:
            sell_price = last_buy_price * 1.10
        # 触达止损点 卖出   止损设置在-7%
        sell_flag_1 = r['low'] < last_buy_price * 0.93
        if sell_flag_1:
            sell_price = r['low']
        # 跌破10日线的2% 卖出
        sell_flag_2 = r['low'] < ma_10 * 0.98
        if sell_flag_2:
            sell_price = r['close']


        if last_bs_type == 'B' and (last_buy_date != r['trade_date']):
            if sell_flag_1 or sell_flag_2 or sell_flag_3:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append(
                    {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'},
                    ignore_index=True)

    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'},
            ignore_index=True)

    return bs_df




# todo 倒T是否是底部?


# todo 下跌量能比不上上涨量能


'''
先大阴后大阳 策略

'''
def strategy_big_rise_after_big_fail(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 100:
        return bs_df

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 30)

    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0
    last_buy_date = 0


    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time:
            continue
        if i < 60:
            continue

        # buy
        buy_flag_1 = True
        # 必须先有一个大于5的涨幅
        if stock_df.loc[i-1]['chg'] < -5 and stock_df.loc[i]['chg'] > 5 and stock_df.loc[i]['chg'] < 9:
            buy_flag_1 = True

        if buy_flag_1 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            last_buy_price = r['close']
            last_buy_date = r['trade_date']
            bs_df = bs_df.append(
                {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': r['close'], 'type': 'B'},
                ignore_index=True)

        # sell
        # 收益达到 10% 卖出
        sell_flag_3 = (last_buy_price != 0) and r['high'] > last_buy_price * 1.10
        if sell_flag_3:
            sell_price = last_buy_price * 1.10
        # 触达止损点 卖出   止损设置在-7%
        sell_flag_1 = r['low'] < last_buy_price * 0.93
        if sell_flag_1:
            sell_price = r['low']


        if last_bs_type == 'B' and (last_buy_date != r['trade_date']):
            if sell_flag_1:
                last_bs_type = 'S'
                last_buy_price = 0
                bs_df = bs_df.append(
                    {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': sell_price, 'type': 'S'},
                    ignore_index=True)

    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'price': r['close'], 'type': 'S'},
            ignore_index=True)

    return bs_df





def get_stock_by_index():
    index_list = ['沪深300','中证500','中证800成长','中证800价值','中证1000']

    total_stock_set = set()
    # todo
    # for index in index_list:
    #     with open('D:\\work\\project\\jk_data\\stock_base\\指数及成分\\'+index+'.txt', 'r') as f:
    #         l = f.readline()
    #         weight_list = json.loads(l)
    #         for w in weight_list:
    #             stock_code = w['wind_code'][0:6]
    #             total_stock_set.add(stock_code)

    return total_stock_set





# def handle_stock_strategy(strategy, stocks, index_stock_set):
def handle_stock_strategy(context):
    strategy = context['strategy']
    stocks = context['stocks']
    index_stock_set = context['index_stock_set']
    start_strategy_time = context['start_strategy_time']
    end_strategy_time = context['end_strategy_time']

    cash_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'cash'])
    bs_df_total = pd.DataFrame(columns=['stock_code', 'rcd_date', 'close', 'price', 'type'])

    i = 1
    for stock in stocks:
        print("handle {}  {}/{} ".format(stock['stock_code'], i, len(stocks)))
        i += 1

        stock_code = stock['stock_code']
        stock_name = stock['stock_name']

        if stock_code.endswith('.BJ'):
            continue

        # 筛掉不关注得股票
        # if not stock_code[0:6] in index_stock_set:
        #     continue

        stock_df = get_stock_daily_price_as_df(stock_code)
        # stock_df.set_index(["rcd_date"], inplace=True)

        # 构建买卖点
        bs_df = strategy(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time)
        # 按买卖点回测
        profit_ratio = bs_test_a(bs_df)
        cash_df = cash_df.append({'stock_code': stock_code, 'stock_name': stock_name, 'profit_ratio': profit_ratio},
                                 ignore_index=True)
        bs_df_total = pd.concat([bs_df_total, bs_df], axis=0)

    context['cash_df'] = cash_df
    context['bs_df_total'] = bs_df_total
    return cash_df, bs_df_total



save_dir = './result/strategy/'
# save_dir = 'D:\\work\\project\\jk_data\\strategy_result\\'
def filter_strategy(strategy, strategy_name, start_strategy_time, end_strategy_time):

    stocks = get_all_stock()
    index_stock_set = get_stock_by_index()

    cash_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'cash'])
    bs_df_total = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])


    start_idx = 0
    pool = mp.Pool(processes = 10)
    ctx_list = []
    while start_idx < len(stocks):
        end_idx = min(start_idx + 500, len(stocks))
        ctx = {"strategy":strategy,"stocks":stocks[start_idx: end_idx],"index_stock_set":index_stock_set, "start_strategy_time":start_strategy_time, "end_strategy_time":end_strategy_time}
        ctx_list.append(ctx)
        # ret = pool.apply_async(handle_stock_strategy, (strategy, stocks[start_idx: end_idx], index_stock_set,))
        # ret_list.append(ret)

        # p = multiprocessing.Process(target=handle_stock_strategy,
        #                             args=(strategy, stocks[start_idx: end_idx], index_stock_set, ))
        # p.start()
        start_idx = start_idx + 500

    rets = pool.map(handle_stock_strategy, ctx_list)

    # pool.close()
    # pool.join()

    for ret in rets:
        cash_df = pd.concat([cash_df, ret[0]], axis=0)
        bs_df_total = pd.concat([bs_df_total, ret[1]], axis=0)

    cash_df.sort_values(by = 'profit_ratio')

    cash_df.to_csv(save_dir+strategy_name+'_cash_df.csv', index=False, sep=',', encoding='utf-8-sig')
    bs_df_total.to_csv(save_dir+strategy_name+'_bs_df_total.csv', index=False, sep=',', encoding='utf-8-sig')

    # 结果输出
    print('total_profit_ratio {}'.format({cash_df['profit_ratio'].sum()}))
    print('total_profit_ratio {}'.format({cash_df['profit_ratio'].sum()}))


    return bs_df



def differ_stock():
    dfa = pd.read_csv('D:\\work\\project\\jk_data\\strategy_result\\a.csv')
    dfb = pd.read_csv('D:\\work\\project\\jk_data\\strategy_result\\b.csv')
    df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    stock_set = set()
    for i in dfa.index:
        stock = dfa.loc[i]['stock_code']
        stock_set.add(stock)

    for i in dfb.index:
        stock = dfb.loc[i]['stock_code']
        if stock in stock_set:
            continue
        else:
            df = df.append(dfb.loc[i], ignore_index=True)
            # final_df.append({'stock_code': index, 'S': S, 'K': K}, ignore_index=True)
    df.to_csv('D:\\work\\project\\jk_data\\strategy_result\\c.csv', index=False, sep=',', encoding='utf-8-sig')


if __name__ == '__main__':
    differ_stock()

    # get_stock_by_index()

    # 指定日期范围
    start_strategy_time = datetime.date(2022, 10, 12)
    end_strategy_time = datetime.date(2022, 10, 13)

    # 龙头战法 仅供研究用
    # filter_strategy(strategy_first_fall_in_successive_up, '研究策略\\龙头战法', start_strategy_time, end_strategy_time)
    # 缩量大跌 仅供研究用
    # filter_strategy(strategy_fall_with_high_chg_and_low_vol, '研究策略\\缩量大跌', start_strategy_time, end_strategy_time)

    # 上穿20日线
    filter_strategy(strategy_more_than_ma_30, '上穿20日线', start_strategy_time, end_strategy_time)
    # 大阳后小幅回调
    # filter_strategy(strategy_after_big_increase, '大阳回调', start_strategy_time, end_strategy_time)
    # 交易量 阳大阴小
    filter_strategy(strategy_rise_high_vol_vs_down_low_vol, '交易量阳大阴小', start_strategy_time, end_strategy_time)
    # 小阳堆积
    # filter_strategy(strategy_successive_small_up, '小阳堆积', start_strategy_time, end_strategy_time)
    # 先大阴后大阳
    # filter_strategy(strategy_big_rise_after_big_fail, '先大阴后大阳', datetime.date(2021, 3, 23), datetime.date(2022, 6, 30))
