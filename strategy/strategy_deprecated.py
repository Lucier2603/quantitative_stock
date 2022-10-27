

import pandas as pd

from strategy.assist import ma_i

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





'''
先大阴后大阳 策略

'''
def strategy_big_rise_after_big_fail(stock_code, stock_name, stock_df, start_strategy_time, end_strategy_time):
    # 买卖记录点
    bs_df = pd.DataFrame(columns=['stock_code', 'stock_name', 'trade_date', 'close', 'price', 'type'])

    if len(stock_df) < 100:
        return bs_df

    for i in stock_df.index:
        r = stock_df.loc[i]

        # 从指定日期开始
        if r['trade_date'] < start_strategy_time:
            continue
        if i < 60:
            continue

        # buy
        buy_flag_1 = False
        # 必须先有一个大于5的涨幅
        if stock_df.loc[i-1]['chg'] < -5 and stock_df.loc[i]['chg'] > 5 and stock_df.loc[i]['chg'] < 9:
            buy_flag_1 = True

        if buy_flag_1:
            bs_df = bs_df.append(
                {'stock_code': stock_code, 'stock_name':stock_name, 'trade_date': r['trade_date'], 'price': r['close'], 'type': 'B'},
                ignore_index=True)

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





