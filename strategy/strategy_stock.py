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

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time or i < 30 or r['trade_date'] > end_strategy_time:
            continue

        ma5 = ma_i(stock_df, 'close', 5, i)
        ma10 = ma_i(stock_df, 'close', 10, i)
        ma20 = ma_i(stock_df, 'close', 20, i)

        # 买入条件 1.  ma5 > ma10
        if ma5 < ma10:
            continue

        # 买入条件 2.  ma10从刚刚下向上穿3天; ma10在ma20下运行至少5日; 收盘价均高于ma5的98%
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

        # 买入条件 1. 5日内价格必须在最高的10%以上
        # todo 这个应该作为寻找趋势的策略
        # highest_5 = 0
        # for j in range(1, 6):
        #     rr = stock_df.loc[i-j]
        #     if rr['close']>highest_5:
        #         highest_5 = rr['close']
        # if r['close'] < highest_5*0.95:
        #     continue


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
    # ma(stock_df, 'close', 10)
    # ma(stock_df, 'close', 20)
    # ma(stock_df, 'close', 30)
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

        buy_flag_1 = True

        # 买入条件 1. 10日内的最高价都在10日线上  2. 10日线必须都在20日线上
        fail_falg = False
        for j in range(0, 12):
            r2 = stock_df.loc[i-j]
            if r2['high'] < ma_i(stock_df, 'close', 10, i-j):
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
        #     if r2['chg'] > 0:
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
行业共振 策略
'''
def strategy_industry_top(stock_code, stock_df):
    return 1











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
    bs_df_total.to_csv(save_dir + strategy_name + '_bs_df_total.csv', index=False, sep=',', encoding='utf-8-sig')
    bs_df_total = bs_df_total[bs_df_total['trade_date']<end_strategy_time]
    bs_df_total.to_csv(save_dir + strategy_name + '_bs_df_add.csv', index=False, sep=',', encoding='utf-8-sig')

    # bs_df_total.to_csv(save_dir+strategy_name+'_bs_df_total.csv', index=False, sep=',', encoding='utf-8-sig')




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
