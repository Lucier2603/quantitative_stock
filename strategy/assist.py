import scipy.stats as stats
import numpy as np
from pyecharts import options
from scipy.stats.stats import _contains_nan
from pyecharts.charts import Line, Bar

'''
均线
'''
def ma(df, col_name, n, start_date=None):
    ma_col_name = col_name+'_ma'+str(n)
    df[ma_col_name] = 0

    for i in df.index:
        r = df.loc[i]
        if i < n:
            # r[ma_col_name] = 0
            df.loc[i, ma_col_name] = 0
        else:
            if start_date is not None and r['trade_date']<start_date:
                df.loc[i, ma_col_name] = 0
            else:
                total = 0
                for j in range(i + 1 - n, i + 1):
                    total += df.loc[j, col_name]
                # r[ma_col_name] = total / n
                df.loc[i, ma_col_name] = total / n


'''
均线 返回单值均线
返回当前第i日/行，向前n个记录的ma_n值
'''
def ma_i(df, col_name, n, i):
    total = 0
    for j in range(0, n):
        total += df.loc[i-j, col_name]

    return total / n



'''
zscore 返回单值zscore

返回以n日作为周期的zscore
'''
def zscore_price_i(df):
    df['zscore'] = stats.zscore(df['close'])

    a = df['close']
    a = np.asanyarray(a)

    contains_nan, nan_policy = _contains_nan(a, 'propagate')

    if contains_nan and nan_policy == 'omit':
        mns = np.nanmean(a=a, keepdims=True)
        sstd = np.nanstd(a=a, keepdims=True)
    else:
        mns = a.mean(keepdims=True)
        sstd = a.std(keepdims=True)

    df['zscore'] = (a - mns) / np.sqrt(sstd)
    # return (a - mns) / sstd
    return df


'''
ATR
'''
def cal_atr(df, n):
    for i in range(0, len(df)):
        df.loc[df.index[i], 'TR'] = max((df['close'][i] - df['low'][i]), (df['high'][i] - df['close'].shift(-1)[i]),
                                        (df['low'][i] - df['close'].shift(-1)[i]))
    df['ATR'] = df['TR'].rolling(n).mean()
    return df


'''
日线实体相关
'''
# 上影线与实体比
# 这里使用上影线 比上 带下影线的实体
def high_vs_close(high, open, close, low):
    if open>close:
        a=open
        b=close
    else:
        a = close
        b = open
    return abs( (high-a)/(a-low) ) if a!=low else 99


def chg(before, after):
    return after/before-1



'''
bs_df收益回测
本金10w
每个B买满
每个S清空卖出所有
'''
def bs_test_a(bs_df):
    # 当前持有现金
    cash = 100000
    # 当前持有股票
    hold_stock = 0

    for i in bs_df.index:
        r = bs_df.loc[i]
        if r['type'] == 'B' and hold_stock == 0:
            hold_stock = ((cash / r['price']) // 100) * 100
            cash -= hold_stock * r['price']
        elif r['type'] == 'S' and hold_stock != 0:
            cash += hold_stock * r['price']
            hold_stock = 0

    return round((cash-100000)/1000, 2)










'''
画图
'''
def to_line_chart(x_name, x, y_map):
    chart = Line(init_opts=options.InitOpts(width="1800px", height="1200px"))
    chart.add_xaxis(x)

    for y_name in y_map.keys():
        chart.add_yaxis(y_name, y_map[y_name])

    # line.render_notebook()
    chart.render('line.html')

def to_bar_chart(chart_name, x, y_map):
    chart = Bar(init_opts=options.InitOpts(width="1800px", height="1200px"))
    chart.add_xaxis(x)

    for y_name in y_map.keys():
        chart.add_yaxis(y_name, y_map[y_name])

    # line.render_notebook()
    chart.render('D:\\work\\量化\\' + chart_name + '.html')