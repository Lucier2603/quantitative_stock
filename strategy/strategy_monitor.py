import datetime

from numpy import std

from strategy.assist import ma, chg, zscore_price_i, to_line_chart, to_bar_chart
from ts.stock_data_service import get_all_etf, get_etf_daily_price_as_df, get_index_daily_price_as_df, get_all_stock, \
    get_stock_daily_price_as_df, get_all_index, get_stock_info

from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB


'''
ETF趋势监测
方案：zscore
'''
def etf_trend_monitor(start_time):
    etfs = get_all_etf()

    # 获取 000985.CSI 中证全指 作为基准
    basic_df = get_index_daily_price_as_df('000985.CSI', 'CSI')

    i = 1

    # 20日内涨幅最好的
    chg_20_day_list = []
    for etf in etfs:
        print('handle {} {}/{}'.format(etf.stock_code, i, len(etfs)))
        i += 1

        etf_df = get_etf_daily_price_as_df(etf.stock_code)

        if len(etf_df) < 90:
            continue
        if etf.stock_name.find('债')>0 or etf.stock_name.find('货币')>0:
            continue

        # 方案2 计算最近30日的zscore
        score = zscore_monitor(etf_df)
        chg_20_day_list.append({'etf_code': etf.stock_code, 'etf_name': etf.stock_name, 'score': score})

    chg_20_day_list.sort(key=lambda x: x['score'], reverse=True)
    chg_20_day_list = [x for x in chg_20_day_list if x['score']>0.3]

    for d in chg_20_day_list:
        print('{} {} --- {}'.format(d['etf_code'],d['etf_name'],round(d['score']*100)))


'''
指数趋势监测
方案：zscore
'''
def index_trend_monitor(start_time):
    indexes = get_all_index()

    # 获取 000985.CSI 中证全指 作为基准
    basic_df = get_index_daily_price_as_df('000985.CSI', 'CSI')

    i = 1

    # 20日内涨幅最好的
    chg_20_day_list = []
    for index in indexes:
        print('handle {} {}/{}'.format(index.index_code, i, len(indexes)))
        i += 1

        etf_df = get_index_daily_price_as_df(index.index_code, index.market)

        if len(etf_df) < 90:
            continue

        # 方案2 计算最近30日的zscore
        score = zscore_monitor(etf_df)
        chg_20_day_list.append({'etf_code': index.index_code, 'etf_name': index.index_name, 'score': score})

    chg_20_day_list.sort(key=lambda x: x['score'], reverse=True)
    chg_20_day_list = [x for x in chg_20_day_list if x['score']>0.3]

    for d in chg_20_day_list:
        print('{} {} --- {}'.format(d['etf_code'],d['etf_name'],round(d['score']*100)))



'''
三大指数 均线偏离度
'''
def main_index_ma_deviation(start_time):

    index_df = get_index_daily_price_as_df('000905.SH', 'SSE')

    index_df = index_df[index_df['trade_date']>start_time]
    index_df.reset_index(inplace=True)
    index_df = index_df[['trade_date','close','low']]

    # 计算均线
    ma(index_df, 'close', 5)
    # 计算偏离度
    index_df['dev_5'] = (index_df['low'] / index_df['close_ma5'] - 1) * 100
    # 筛选偏离度 小于-3 的
    index_df = index_df[index_df['dev_5'] < -3]

    print(1)




'''
股票趋势监测
方案：zscore
'''
def stock_trend_monitor(start_time):
    stocks = get_all_stock()

    # 获取 000985.CSI 中证全指 作为基准
    basic_df = get_index_daily_price_as_df('000985.CSI', 'CSI')

    i = 1

    # 20日内涨幅最好的
    chg_20_day_list = []
    for stock in stocks:
        print('handle {} {}/{}'.format(stock.stock_code, i, len(stocks)))
        i += 1

        etf_df = get_stock_daily_price_as_df(stock.stock_code)

        if len(etf_df) < 90:
            continue

        score = zscore_monitor(etf_df)
        chg_20_day_list.append({'etf_code': stock.stock_code, 'etf_name': stock.stock_name, 'score': score})

    chg_20_day_list.sort(key=lambda x: x['score'], reverse=True)
    chg_20_day_list = [x for x in chg_20_day_list if x['score']>0.3]

    for d in chg_20_day_list:
        print('{} {} --- {}'.format(d['etf_code'],d['etf_name'],round(d['score']*100)))






'''
zscore趋势监测
'''
def zscore_monitor(df):
    n = 45

    df20 = df.tail(n).copy()
    df20.reset_index(drop=True, inplace=True)
    df20 = zscore_price_i(df20)

    # 按权重递减 计算一个总分
    score = 0
    a = 0.01
    for j in df20.index:
        b = 1 - a * (n - j)
        score += df20.loc[j]['zscore'] * b

    return score


def chg_monitor(df):
    print(1)


# 方案1 计算最近5,10,20日的涨幅 并塞入list
# df20 = etf_df.tail(20).copy()
# df20.reset_index(drop=True,inplace=True)
# day5_chg = chg(df20.loc[15]['close'], df20.loc[19]['close'])
# day10_chg = chg(df20.loc[10]['close'], df20.loc[19]['close'])
# day20_chg = chg(df20.loc[0]['close'], df20.loc[19]['close'])
#
# score = day5_chg*0.4+day10_chg*0.25+day20_chg*0.15
# chg_20_day_list.append({'etf_code':etf.stock_code,'etf_name':etf.stock_name,'chg':score})




# 每日创20日新低 vs 20日新高
def highest_lowest(start_time):
    stocks = get_all_stock()

    high_cnt_map = {}
    low_cnt_map = {}

    i = 1
    for stock in stocks:
        print('handle {} {}/{}'.format(stock.stock_code, i, len(stocks)))
        i += 1

        stock_df = get_stock_daily_price_as_df(stock.stock_code)

        stock_df = stock_df[stock_df['trade_date'] > start_time]
        stock_df.reset_index(inplace=True)

        if len(stock_df) < 90:
            continue

        list = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
        for j in stock_df.index:
            r = stock_df.loc[j]

            idx = j % 20
            # 最近20日的close合集
            list[idx] = r['close']

            if j <= 20:
                continue

            min_close = min(list)
            max_close = max(list)

            d = r['trade_date'].strftime("%Y-%m-%d")
            if min_close == r['close']:
                low_cnt_map[d] = low_cnt_map.get(d,0) + 1
            if max_close == r['close']:
                high_cnt_map[d] = high_cnt_map.get(d,0) + 1

    # 输出统计结果
    cur_date = start_time
    x = []
    high = []
    low = []
    diff = []
    while cur_date <= datetime.date.today():
        d = cur_date.strftime("%Y-%m-%d")

        h = high_cnt_map.get(d, 0)
        l = low_cnt_map.get(d, 0)

        if h+l != 0:
            x.append(d)
            high.append(h)
            low.append(l)
            diff.append(h-l)

        cur_date = cur_date + datetime.timedelta(days=1)

    # y_map = {'high': high, 'low': low}
    y_map = {'diff': diff}
    to_bar_chart('新高新低比', x, y_map)


# 计算波动率(20日)
def volatility(stock_codes, start_time):
    for stock_code in stock_codes:
        stock = get_stock_info(stock_code)
        stock_name = stock[0].stock_name

        stock_df = get_stock_daily_price_as_df(stock_code)

        stock_df = stock_df[stock_df['trade_date'] > start_time]
        stock_df.reset_index(inplace=True)

        list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dates = []
        stds = []
        for j in stock_df.index:
            if j == 0:
                continue

            r1 = stock_df.loc[j - 1]
            r = stock_df.loc[j]

            idx = j % 20
            # 取最大波动
            v = max(abs(r['high'] - r['low']), abs(r1['close'] - r['low']), abs(r1['close'] - r['high']))
            v = v / r1['close']
            list[idx] = v

            if j <= 20:
                continue

            # 计算标准差
            dates.append(r['trade_date'].strftime("%Y-%m-%d"))
            stds.append(std(list))

        y_map = {'std': stds}
        to_bar_chart('波动率\\' + stock_name, dates, y_map)









if __name__ == '__main__':
    # 指定日期范围
    start_strategy_time = datetime.date(2020, 7, 1)
    end_strategy_time = datetime.date(2022, 5, 25)

    # etf_trend_monitor(start_strategy_time)
    # index_trend_monitor(start_strategy_time)
    # stock_trend_monitor(start_strategy_time)

    # 每日创新高新低比值
    # highest_lowest(start_strategy_time)
    # 计算指定股票波动率
    volatility(['300750.SZ'], start_strategy_time)



    # 主要股指 ma均线偏离度
    # main_index_ma_deviation(start_strategy_time)
