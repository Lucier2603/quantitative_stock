# coding=utf-8


import time

from engine import stock_engine, TAMP_SQL
from ts.stock_data_service import get_all_stock_code, delete_stock_daily_price, delete_stock_index, \
    delete_stock_index_rel, get_all_index, delete_index_daily_price, get_all_etf, delete_etf_daily_price
from ts.stock_model import BasicStockInfo, StockTradeDaily00x, StockTradeDaily30x, \
    StockTradeDaily60x, StockTradeDaily83x, StockTradeDaily688, StockIndex, StockIndexRel, IndexTradeDailyCSI, \
    IndexTradeDailySSE, IndexTradeDailySZSE, BasicETFInfo, ETFTradeDaily
import pandas as pd
import datetime
import tushare



'''
更新每日股票价格信息 ts
'''
# 更新每日
def update_month_stock_price_ts(p_start_date, p_end_date):
    stocks = get_all_stock_code()

    i = 1
    for stock_code in stocks:
        print('{} start {}/{}'.format(stock_code, i, len(stocks)))
        i += 1

        df = pro.daily(ts_code=stock_code, start_date=p_start_date, end_date=p_end_date)
        do_save_stock_price_ts(stock_code, df)

        time.sleep(0.2)

    print(1)

def do_save_stock_price_ts(stock_code, df):
    engine = stock_engine

    if df is None or df.empty:
        return

    for i in df.index:
        try:
            rcd = df.loc[i]
            trade_date = rcd['trade_date']

            delete_stock_daily_price(stock_code, trade_date)
            ret = {'stock_code': stock_code, 'trade_date': trade_date, 'open': rcd['open'], 'close': rcd['close'],
                   'high': rcd['high'], 'low': rcd['low'], 'vol': rcd['vol'], 'chg': rcd['pct_chg'], 'amount': rcd['amount']}
            with TAMP_SQL(engine) as tamp_fund:
                tamp_fund_session = tamp_fund.session
                if stock_code.startswith('00'):
                    tamp_fund_session.execute(StockTradeDaily00x.__table__.insert(), [ret])
                if stock_code.startswith('30'):
                    tamp_fund_session.execute(StockTradeDaily30x.__table__.insert(), [ret])
                if stock_code.startswith('60'):
                    tamp_fund_session.execute(StockTradeDaily60x.__table__.insert(), [ret])
                if stock_code.startswith('83'):
                    tamp_fund_session.execute(StockTradeDaily83x.__table__.insert(), [ret])
                if stock_code.startswith('68'):
                    tamp_fund_session.execute(StockTradeDaily688.__table__.insert(), [ret])

        except Exception as e:
            print('error in {}'.format(stock_code))
            print(e)




'''
更新每日指数价格信息 ts
'''
# 更新每日
def update_daily_index_price(p_start_date, p_end_date):
    indexes = get_all_index()

    i = 1
    for index in indexes:
        index_code = index.index_code
        print('{} start {}/{}'.format(index_code, i, len(indexes)))
        i += 1

        if i <= 1032:
            continue

        df = pro.index_daily(ts_code=index_code, start_date=p_start_date, end_date=p_end_date)
        do_save_index_price_ts(index_code, index.market, df)

        time.sleep(0.2)


def do_save_index_price_ts(index_code, market, df):
    engine = stock_engine

    if df is None or df.empty:
        return

    for i in df.index:
        try:
            rcd = df.loc[i]
            trade_date = rcd['trade_date']

            delete_index_daily_price(index_code, market, trade_date)
            ret = {'index_code': index_code, 'trade_date': trade_date, 'open': rcd['open'], 'close': rcd['close'],
                   'high': rcd['high'], 'low': rcd['low'], 'vol': rcd['vol'], 'chg': rcd['pct_chg'], 'amount': rcd['amount']}
            with TAMP_SQL(engine) as tamp_fund:
                tamp_fund_session = tamp_fund.session
                if market == 'CSI':
                    tamp_fund_session.execute(IndexTradeDailyCSI.__table__.insert(), [ret])
                if market == 'SSE':
                    tamp_fund_session.execute(IndexTradeDailySSE.__table__.insert(), [ret])
                if market == 'SZSE':
                    tamp_fund_session.execute(IndexTradeDailySZSE.__table__.insert(), [ret])


        except Exception as e:
            print('error in {}'.format(index_code))
            print(e)



'''
更新每日ETF信息 ts
https://tushare.pro/document/2?doc_id=127
'''
# 更新每日
def update_daily_etf_price(p_start_date, p_end_date):
    engine = stock_engine
    etfs = get_all_etf()

    j = 1
    for etf in etfs:
        etf_code = etf.stock_code
        print('{} start {}/{}'.format(etf_code, j, len(etfs)))
        j += 1

        df = pro.fund_daily(ts_code=etf_code, start_date=p_start_date, end_date=p_end_date)

        # save
        if df is None or df.empty:
            return

        for i in df.index:
            try:
                rcd = df.loc[i]
                trade_date = rcd['trade_date']
                delete_etf_daily_price(etf_code, trade_date)
                ret = {'stock_code': etf_code, 'trade_date': trade_date, 'open': rcd['open'], 'close': rcd['close'],
                   'high': rcd['high'], 'low': rcd['low'], 'vol': rcd['vol'], 'chg': rcd['pct_chg'],
                   'amount': rcd['amount']}
                with TAMP_SQL(engine) as tamp_fund:
                    tamp_fund_session = tamp_fund.session
                    tamp_fund_session.execute(ETFTradeDaily.__table__.insert(), [ret])

            except Exception as e:
                print('error in {}'.format(etf_code))
                print(e)

        time.sleep(0.2)


def do_save_index_price_ts(index_code, market, df):
    engine = stock_engine

    if df is None or df.empty:
        return

    for i in df.index:
        try:
            rcd = df.loc[i]
            trade_date = rcd['trade_date']

            delete_index_daily_price(index_code, market, trade_date)
            ret = {'index_code': index_code, 'trade_date': trade_date, 'open': rcd['open'], 'close': rcd['close'],
                   'high': rcd['high'], 'low': rcd['low'], 'vol': rcd['vol'], 'chg': rcd['pct_chg'], 'amount': rcd['amount']}
            with TAMP_SQL(engine) as tamp_fund:
                tamp_fund_session = tamp_fund.session
                if market == 'CSI':
                    tamp_fund_session.execute(IndexTradeDailyCSI.__table__.insert(), [ret])
                if market == 'SSE':
                    tamp_fund_session.execute(IndexTradeDailySSE.__table__.insert(), [ret])
                if market == 'SZSE':
                    tamp_fund_session.execute(IndexTradeDailySZSE.__table__.insert(), [ret])


        except Exception as e:
            print('error in {}'.format(index_code))
            print(e)







'''
更新指数成分 月度
see https://tushare.pro/document/2?doc_id=94
see https://tushare.pro/document/2?doc_id=96
'''
def sync_all_index():
    engine = stock_engine

    market_list = ['CSI', 'SSE', 'SZSE']

    for p_market in market_list:
        index_df = pro.index_basic(market=p_market)

        for i in index_df.index:
            print("start {} {}/{}".format(p_market, i, len(index_df)))

            if i < 0:
                continue

            r = index_df.loc[i]
            stock_df = pro.index_weight(index_code=r['ts_code'], start_date='20220420', end_date='20220505')

            if stock_df.empty:
                time.sleep(0.5)
                continue

            delete_stock_index(r['ts_code'])
            delete_stock_index_rel(r['ts_code'])

            stock_weight = {}
            stock_index_rels = []
            for j in stock_df.index:
                s = stock_df.loc[j]
                stock_weight[s['con_code']] = s['weight']
                trade_date = s['trade_date']
                stock_index_rels.append({'index_code':r['ts_code'], 'stock_code':s['con_code'], 'weight':s['weight']})

            ret = {'index_code': r['ts_code'], 'index_name': r['name'], 'trade_date': trade_date, 'market': r['market'], 'category': r['category'],
                   'stock_weight': stock_weight}
            with TAMP_SQL(engine) as tamp_fund:
                tamp_fund_session = tamp_fund.session
                tamp_fund_session.execute(StockIndex.__table__.insert(), [ret])
                tamp_fund_session.execute(StockIndexRel.__table__.insert(), stock_index_rels)

            # print("{} done".format(r['ts_code']))
            time.sleep(0.5)

    print(1)


'''
获取所有ETF信息
https://tushare.pro/document/2?doc_id=19
'''
def sync_all_ETF():
    engine = stock_engine

    # df = pro.fund_basic()
    df = pro.fund_basic(market='E', status='L')

    for i in df.index:
        stock = df.loc[i]

        ret = {'stock_code': stock['ts_code'], 'fund_type': stock['fund_type'], 'stock_name': stock['name']}
        with TAMP_SQL(engine) as tamp_fund:
            tamp_fund_session = tamp_fund.session
            tamp_fund_session.execute(BasicETFInfo.__table__.insert(), [ret])


'''
获取所有股票信息
see https://waditu.com/document/2?doc_id=25
'''
def sync_all_stock():
    engine = stock_engine

    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date,list_status,is_hs')

    for i in df.index:
        print('{}/{}'.format(i, len(df)))
        stock = df.loc[i]

        is_hk = 'N' if stock['is_hs'] == 'N' else 'Y'
        ret = {'stock_code': stock['ts_code'], 'stock_symbol': stock['symbol'], 'stock_name': stock['name'], 'area': stock['area'], 'market': stock['market'], 'industry': stock['industry'], 'is_hk': is_hk}
        with TAMP_SQL(engine) as tamp_fund:
            tamp_fund_session = tamp_fund.session
            tamp_fund_session.execute(BasicStockInfo.__table__.insert(), [ret])







if __name__ == '__main__':
    tushare.set_token('702d425922f79bd2ebda78a369f6fe1a40982bbf6a51e00c61bec879')
    pro = tushare.pro_api()


    # sync_all_stock()
    sync_all_index()
    sync_all_ETF()


    p_start_date = '20220907'
    p_end_date = '20220930'

    # 更新股票信息
    # update_month_stock_price_ts(p_start_date, p_end_date)
    # 更新指数信息
    # update_daily_index_price(p_start_date, p_end_date)
    # 更新etf信息
    # update_daily_etf_price(p_start_date, p_end_date)