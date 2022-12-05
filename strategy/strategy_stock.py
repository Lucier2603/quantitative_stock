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

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 20)
    # ma(stock_df, 'close', 30)

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        ma5 = ma_i(stock_df, 'close', 5, i)
        ma10 = ma_i(stock_df, 'close', 10, i)
        ma20 = ma_i(stock_df, 'close', 20, i)

        # 买入条件 1. ma5 > ma10
        if ma5 < ma10:
            continue

        # 买入条件 2. ma10从刚刚下向上穿3天; ma10在ma20下运行至少5日; 收盘价均高于ma5的98%
        if ma10 < ma20:
            continue
        buy_flag_1 = True
        for j in range(3, 8):
            ma10j = ma_i(stock_df, 'close', 10, i-j)
            ma20j = ma_i(stock_df, 'close', 20, i-j)
            if ma10j > ma20j:
                buy_flag_1 = False
                break
        if not buy_flag_1:
            continue
        for j in range(1, 8):
            # 收盘价均高于ma5的98%
            close5j = stock_df.loc[i - j, 'close']
            ma5j = ma_i(stock_df, 'close', 5, i - j)
            if close5j < ma5j * 0.985:
                buy_flag_1 = False
                break
            # todo 控制一下下跌日子数量
        if not buy_flag_1:
            continue


        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name': stock_name, 'trade_date': r['trade_date']}, ignore_index=True)

    return bs_df


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

    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 20)
    # ma(stock_df, 'close', 30)

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        ma5 = ma_i(stock_df, 'close', 5, i)
        ma10 = ma_i(stock_df, 'close', 10, i)
        ma20 = ma_i(stock_df, 'close', 20, i)

        # 买入条件 1. ma5 > ma10
        if ma5 < ma10:
            continue

        # 买入条件 2. ma10从刚刚下向上穿3天; ma10在ma20下运行至少5日; 收盘价均高于ma5的98%
        if ma10 < ma20:
            continue
        buy_flag_1 = True
        for j in range(3, 8):
            ma10j = ma_i(stock_df, 'close', 10, i-j)
            ma20j = ma_i(stock_df, 'close', 20, i-j)
            if ma10j > ma20j*0.98:
                buy_flag_1 = False
                break
        if not buy_flag_1:
            continue
        for j in range(1, 8):
            # 收盘价均高于ma5的98%
            close5j = stock_df.loc[i - j, 'close']
            ma5j = ma_i(stock_df, 'close', 5, i - j)
            if close5j < ma5j * 0.985:
                buy_flag_1 = False
                break
            # todo 控制一下下跌日子数量
        if not buy_flag_1:
            continue


        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name': stock_name, 'trade_date': r['trade_date']}, ignore_index=True)


    return bs_df



'''
寻找趋势 策略
'''
def strategy_find_trend_A(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df

    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()

    #
    N = 20
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
            if (r2['close'] < ma_i(stock_df, 'close', 30, i)) or (r2['close'] < ma_i(stock_df, 'close', 20, i)):
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
                # bs_df = bs_df.append({'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'close': r['close'], 'price': r['close'], 'type': 'S'}, ignore_index=True)


    # 最后一天作为卖出
    if last_bs_type == 'B':
        r = stock_df.iloc[-1]
        # bs_df = bs_df.append(
        #     {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': '--', 'close': r['close'], 'price': r['close'],
        #      'type': 'N'}, ignore_index=True)

    return bs_df



'''
寻找趋势 策略
'''
def strategy_find_trend_B(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
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

        ma_5 = ma_i(stock_df, 'close', 5, i)
        ma_10 = ma_i(stock_df, 'close', 10, i)
        ma_20 = ma_i(stock_df, 'close', 20, i)


        buy_flag_1 = True
        # 买入条件 1. 10日内的最高价都在10日线上  2. 10日线必须都在20日线上
        fail_falg = False
        for j in range(0, 12):
            r2 = stock_df.loc[i-j]
            if r2['high'] < ma_i(stock_df, 'close', 10, i-j) * 0.985:
                fail_falg = True
                break
            if ma_i(stock_df, 'close', 10, i-j) < ma_i(stock_df, 'close', 20, i-j):
                fail_falg = True
                break
        if fail_falg:
            continue

        # 买入条件 2. 须在60内的20%区间上
        # max_60 = stock_df.rolling(60).max()
        # if r['close'] < max_60 * 0.8:
        #     continue

        # 买入条件 3. 5日线必须高于20日线至少5个百分点
        if ma_5<ma_20*1.05:
            continue

        # 买入条件 4. 拒绝大阴线 收阴且open-close大于7%
        buy_flag_4 = True
        for j in range(0, 12):
            r2 = stock_df.loc[i - j]
            if r2['open']/r2['close']>1.07:
                buy_flag_4 = False
                break
        if not buy_flag_4:
            continue

        # 买入条件 5. 成交量最大的一天(大于1.5倍红色平均值) 不可以收阴线
        max_vol = 0
        red_total_vol = 0
        red_days = 0
        is_red = True
        for j in range(0, 12):
            r2 = stock_df.loc[i - j]
            if r2['close'] > r2['open']:
                red_total_vol += r2['vol']
                red_days += 1

            if r2['vol'] > max_vol:
                max_vol = r2['vol']
                is_red = r2['close'] > r2['open']
        if red_days == 0:
            continue
        avg_red_vol = red_total_vol / red_days
        if (max_vol > avg_red_vol * 1.5) and (not is_red):
            continue

        # 买入条件 6. RSI强度指数 阴线阳线数量比
        # red_cnt = 0
        # green_cnt = 0
        # for j in range(0, 12):
        #     r2 = stock_df.loc[i - j]
        #     if r2['close'] > r2['open']:
        #         red_cnt += 1
        #     else:
        #         green_cnt += 1
        # if red_cnt < (green_cnt*1.5):
        #     continue

        # 买入条件 7. 在5日线上2%以内，10日线上+-3% 20日线上+-3%
        buy_flag_2 = False

        if r['close']<ma_5*1.02 and r['close']>ma_5*0.98:
            buy_flag_2 = True
        if r['close']<ma_10*1.03 and r['close']>ma_10*0.97:
            buy_flag_2 = True
        if r['close']<ma_20*1.03 and r['close']>ma_20*0.97:
            buy_flag_2 = True
        if not buy_flag_2:
            continue


        # 买入条件 8. 当日收阴
        if r['close'] > r['open']:
            continue

        # 买入条件 9. 无6%以上暴跌
        buy_flag_3 = True
        for j in range(0, 10):
            r2 = stock_df.loc[i - j]
            if r2['chg'] < -6:
                buy_flag_3 = False
                break
        if not buy_flag_3:
            continue

        # todo 买入条件 9. N日内处于20%区间 非常重要！！


        # 止损条件 1. 止损时点：ma20和ma30中间，目标是防止大阴线
        # 止损条件 2. 止损时点：最高价向下10%
        # 卖出条件 3. 频繁长上影线

        if buy_flag_1:
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'close':r['open'], 'price':r['open'], 'type':'B'}, ignore_index=True)

    return bs_df



'''
回调到ma均线 策略
'''
def strategy_back_to_ma(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df
    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()


    # 上一个交易行为 买入or卖出 用以防止连续买入or连续卖出
    last_bs_type = 'N'
    # 上一个买入价格 用以止损点设置
    last_buy_price = 0


    # todo 参考601628
    # todo 必须先有一定的涨幅

    for i in stock_df.index:
        r = stock_df.loc[i]
        cur_date = r['trade_date']

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        buy_flag_1 = True

        # 买入条件 1. 接近ma10 ma20 ma30  2. 缩量
        fail_falg = False
        for j in range(0, 12):
            r2 = stock_df.loc[i-j]
            if r2['high'] < ma_i(stock_df, 'close', 10, i-j) * 0.985:
                fail_falg = True
                break
            if ma_i(stock_df, 'close', 10, i-j) < ma_i(stock_df, 'close', 20, i-j):
                fail_falg = True
                break
        if fail_falg:
            continue

        # 买入条件 2. 须在60内的20%区间上
        # max_60 = stock_df.rolling(60).max()
        # if r['close'] < max_60 * 0.8:
        #     continue

        # 买入条件 3. 5日线和10日线，不能一直纠缠，需要有一定差距

        # todo 买入条件 4. 拒绝大阴线 收阴且open-close大于5%

        # 买入条件 5. 成交量最大的一天(大于2倍平均值) 不可以收阴线
        max_vol = 0
        total_vol = 0
        is_red = True
        for j in range(0, 12):
            r2 = stock_df.loc[i - j]
            total_vol += r2['vol']
            if r2['vol'] > max_vol:
                max_vol = r2['vol']
                is_red = r2['chg'] > 0
        avg_vol = total_vol / 12
        if (max_vol > avg_vol * 2) and (not is_red):
            continue

        # 买入条件 6. RSI强度指数 阴线阳线数量比
        # red_cnt = 0
        # green_cnt = 0
        # for j in range(0, 12):
        #     r2 = stock_df.loc[i - j]
        #     if r2['close'] > r2['open']:
        #         red_cnt += 1
        #     else:
        #         green_cnt += 1
        # if red_cnt < (green_cnt*1.5):
        #     continue


        # 止损条件 1. 止损时点：ma20和ma30中间，目标是防止大阴线
        # 止损条件 2. 止损时点：最高价向下10%
        # 卖出条件 3. 频繁长上影线

        if buy_flag_1:
            last_bs_type = 'B'
            last_buy_price = r['close']
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date'], 'close':r['open'], 'price':r['open'], 'type':'B'}, ignore_index=True)

    return bs_df



'''
寻找趋势 策略
'''
def strategy_back_to_ma(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) == 0:
        return bs_df
    # 先做4年内筛选
    stock_df = stock_df[stock_df['trade_date'] > (start_strategy_time-datetime.timedelta(days=1000))]
    stock_df = stock_df.reset_index()

    for i in stock_df.index:
        r = stock_df.loc[i]
        cur_date = r['trade_date']

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        # 买入条件 1. 回调到ma10 20 30
        ma5 = ma_i(stock_df, 'close', 5, i)
        ma10 = ma_i(stock_df, 'close', 10, i)
        ma20 = ma_i(stock_df, 'close', 20, i)
        ma30 = ma_i(stock_df, 'close', 30, i)

        diff10 = abs(r['close']/ma10-1)
        diff20 = abs(r['close']/ma20-1)
        diff30 = abs(r['close']/ma30-1)

        if diff10<0.015 or diff20<0.015 or diff30<0.015:
            buy_flag_1 = True
        else:
            continue

        # 买入条件 2. 缩量下跌0.75

        # 当天如果收红且振幅>0.75 则continue
        if r['close']/r['open']-1>0.0075:
            continue

        red_total_vol = 0
        red_total_days = 0
        for j in range(1, 8):
            r2 = stock_df.loc[i - j]
            if r2['chg'] >0:
                red_total_vol+=r2['vol']
                red_total_days+=1
        if red_total_days==0 or r['vol'] > red_total_vol/red_total_days*0.75:
            continue

        # 买入条件 3. 最高close与当前close至少有5%以上价差
        max_close = 0
        for j in range(1, 8):
            r2 = stock_df.loc[i - j]
            max_close=max_close if max_close>r2['close'] else r2['close']
        if max_close/r['close']<1.04:
            continue

        # 买入条件 4. 均线向上 且有一定差距
        if ma20<ma30 or (ma10/ma20<1.02):
            continue

        # 买入条件 5. 无超过5%以上下跌
        fail_flag = False
        for j in range(1, 15):
            r2 = stock_df.loc[i - j]
            if r2['close']/r2['open']-1 < -0.05:
                fail_flag=True
                continue
        if fail_flag:
            continue

        # 买入条件 6. 无高开大阴线
        fail_flag=False
        max_red_vol = 0
        max_green_vol = 0

        for j in range(1, 10):
            r2 = stock_df.loc[i - j]
            if r2['close']>r2['open'] and r2['vol']>max_red_vol:
                max_red_vol = r2['vol']
            elif r2['close']<r2['open'] and r2['vol']>max_green_vol:
                max_green_vol = r2['vol']
        if max_green_vol>max_red_vol:
            continue

        # todo 买入条件 7. 无连续穿越多个支撑点


        bs_df = bs_df.append(
            {'stock_code': stock_code, 'stock_name': stock_name, 'trade_date': r['trade_date']}, ignore_index=True)

    return bs_df







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
            continue

        # 条件2 5天内的vol比较
        rise_total_vol = 0
        rise_days = 0
        down_total_vol = 0
        down_days = 0
        # 最大的交易量 大于0表示上涨 小于0表示下跌
        max_vol = 0
        for j in range(0,5):
            rt = stock_df.loc[i-j]

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

        if down_days > 4 or down_days == 0:
            continue

        if rise_days*down_days!=0 and (down_total_vol / down_days) / (rise_total_vol / rise_days) > 0.7:
            continue

        # 条件3 5天内 最大交易量必须红 最小交易量为绿 并且至少是1.5倍
        max_red_vol = 0
        max_green_vol = 0
        for j in range(0,5):
            rt = stock_df.loc[i-j]
            if rt['close'] > rt['open']:
                max_red_vol = max_red_vol if max_red_vol>rt['vol'] else rt['vol']
            else:
                max_green_vol = max_green_vol if max_green_vol > rt['vol'] else rt['vol']
        if max_red_vol < max_green_vol*2:
            continue

        if buy_flag_1 and buy_flag_2 and buy_flag_3 and (last_bs_type != 'B'):
            last_bs_type = 'B'
            bs_df = bs_df.append({'stock_code':stock_code, 'stock_name':stock_name, 'trade_date':r['trade_date']}, ignore_index=True)

    return bs_df



def strategy_week_3_red(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 10:
        return bs_df

    #  先计算周数
    stock_df['week'] = stock_df['trade_date'].apply(lambda x: x.isocalendar()[1])
    #  直接更新到每周的df
    week_df = pd.DataFrame()

    cur_week = 999
    last_week_close = 0
    # 上上周 用来计算chg
    last_last_week_close = 0
    last_trade_date = None
    last_week_open = 0
    # 上周的交易日数量 用来计算日均vol
    last_week_days = 0
    last_week_vols = 0

    stock_df = stock_df[stock_df['trade_date'] >= start_strategy_time]
    stock_df = stock_df[stock_df['trade_date'] <= end_strategy_time]

    for i in stock_df.index:
        r = stock_df.loc[i]

        if cur_week == r['week']:
            last_week_close = r['close']
            last_trade_date = r['trade_date']
            last_week_days += 1
            last_week_vols += r['vol']
            continue

        # 新的一周  只记录开盘价格
        last_chg = last_week_close / last_last_week_close -1 if last_last_week_close else 0
        last_avg_vol = last_week_vols / last_week_days if last_week_days else 0
        week_df = week_df.append({'week': r['week'], 'trade_date': last_trade_date, 'open': last_week_open, 'close': last_week_close, 'chg': last_chg, 'days': last_week_days, 'avg_vol': last_avg_vol}, ignore_index=True)
        cur_week = r['week']
        last_week_open = r['open']
        last_last_week_close = last_week_close
        last_week_days = 1
        last_week_vols = r['vol']

    week_df.reset_index(inplace = True)

    for i in week_df.index:
        if i <= 10:
            continue

        r = week_df.loc[i]
        r1 = week_df.loc[i-1]
        r2 = week_df.loc[i-2]
        r3 = week_df.loc[i-3]
        r4 = week_df.loc[i-4]

        # 条件1. 前四周最多只能下跌一周
        buy_flag_1 = True
        green_count = 0
        for j in range(1,5):
            rj = week_df.loc[i-j]
            if rj['close']/rj['open'] < 0.995:
                green_count += 1
        if green_count > 1:
            continue

        # 条件2. 本周涨幅应该介于 -12以内
        if r['close']/r1['close'] < 0.88 or r['close']/r1['close'] > 1.05:
            continue

        # 条件2. 前四周 至少有一周,涨幅大于10%  至少有两周,涨幅大于13%
        if r1['chg']<0.1 and r2['chg']<0.1 and r3['chg']<0.1 and r4['chg']<0.1:
            continue
        tmp_chg = 0
        tmp_chg = max(r1['chg']+r2['chg'], tmp_chg)
        tmp_chg = max(r2['chg']+r3['chg'], tmp_chg)
        tmp_chg = max(r3['chg']+r4['chg'], tmp_chg)
        if tmp_chg<0.13:
            continue


        # 条件3 需要在均线上
        # todo 均线拉开差距
        ma5 = ma_i(week_df, 'close', 5, i)
        ma10 = ma_i(week_df, 'close', 10, i)

        if r['close']<ma5*0.99 or ma5<ma10*1.05:
            continue

        # todo 三连阳后 微微阴线 且vol平均值小


        if buy_flag_1:
            bs_df = bs_df.append({'stock_code': stock_code, 'stock_name': stock_name, 'trade_date': r['trade_date']},
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

    # cash_df.to_csv(save_dir+strategy_name+'_cash_df.csv', index=False, sep=',', encoding='utf-8-sig')
    # 和上一日的bs_df的diff 新增部分
    latest_df = pd.read_csv(save_dir + strategy_name + '_bs_df_total.csv')
    latest_set = set()
    for i in latest_df.index:
        r = latest_df.loc[i]
        latest_set.add(r['stock_code'])

    # 写本次文件
    bs_df_total.to_csv(save_dir + strategy_name + '_bs_df_total.csv', index=False, sep=',', encoding='utf-8-sig')
    # 计算差异 生成diff
    # diff_df = pd.DataFrame()
    # bs_df_total = bs_df_total.reset_index()
    # for i in bs_df_total.index:
    #     r = bs_df_total.loc[i]
    #     d=r['trade_date']
    #     key = r['stock_code']
    #     if key not in latest_set:
    #         diff_df = diff_df.append(r)
    # diff_df.to_csv(save_dir + strategy_name + '_bs_df_diff.csv', index=False, sep=',', encoding='utf-8-sig')




    # bs_df_total = bs_df_total[bs_df_total['trade_date']<end_strategy_time]
    # bs_df_total.to_csv(save_dir + strategy_name + '_bs_df_add.csv', index=False, sep=',', encoding='utf-8-sig')

    # bs_df_total.to_csv(save_dir+strategy_name+'_bs_df_total.csv', index=False, sep=',', encoding='utf-8-sig')




    # 结果输出
    # print('total_profit_ratio {}'.format({cash_df['profit_ratio'].sum()}))
    # print('total_profit_ratio {}'.format({cash_df['profit_ratio'].sum()}))


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
