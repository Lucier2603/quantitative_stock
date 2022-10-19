import datetime
import pandas as pd
import ts as ts
import os

from engine import TAMP_SQL, stock_engine

def backup_db(table):
    now = datetime.datetime.now()
    if not os.path.exists('backup'):
        os.makedirs('backup')

    for file in os.listdir('backup'):
        file_date = datetime.datetime.strptime("-".join(file.split('+')[1].split('-')[:3]).strip('-'), "%Y-%m-%d")
        if (now - file_date).days >= 7:
            os.remove(os.path.join('backup', file))

    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    command = rf"mysqldump -u {config[env]['MySQL']['user']} --password={config[env]['MySQL']['password']} -h {config[env]['MySQL']['host']} {config[env]['MySQL']['tamp_product_db']} {table}> backup/{table}+{now}.sql --column-statistics=0 --set-gtid-purged=OFF"
    return os.system(command)



# basic_stock_info
def get_all_stock_code():
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT stock_code FROM basic_stock_info"""
        cur = tamp_fund_session.execute(sql)
        data = cur.fetchall()

        data = list(map(lambda x: x[0], data))
        return data

def get_all_stock():
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT stock_code, stock_name FROM basic_stock_info"""
        cur = tamp_fund_session.execute(sql)
        data = cur.fetchall()

        return data

# stock_daily_price
def delete_stock_daily_price(stock_code, rcd_date):
    table_name = get_stock_price_table_name(stock_code)
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """DELETE FROM {} WHERE stock_code='{}' and trade_date='{}'""".format(table_name, stock_code, rcd_date)
        cur = tamp_fund_session.execute(sql)

def get_stock_daily_price_as_df(stock_code):
    table_name = get_stock_price_table_name(stock_code)
    if table_name is None:
        return pd.DataFrame(columns=['id', 'stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'chg', 'vol', 'amount'])
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT `id`, `stock_code`, `trade_date`, `open`, `high`, `low`, `close`, `chg`, `vol`, `amount` FROM {} WHERE stock_code='{}' order by trade_date asc""".format(table_name, stock_code)
        cur = tamp_fund_session.execute(sql)
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['id', 'stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'chg', 'vol', 'amount']).dropna(how='any')

        return df

def get_stock_price_table_name(stock_code):
    if stock_code.startswith('00'):
        return 'stock_trade_daily_00x'
    if stock_code.startswith('30'):
        return 'stock_trade_daily_30x'
    if stock_code.startswith('60'):
        return 'stock_trade_daily_60x'
    if stock_code.startswith('83'):
        return 'stock_trade_daily_83x'
    if stock_code.startswith('68'):
        return 'stock_trade_daily_688'


# stock_index
def delete_stock_index(index_code):
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """DELETE FROM stock_index where index_code='{}'""".format(index_code)
        tamp_fund_session.execute(sql)

def delete_stock_index_rel(index_code):
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """DELETE FROM stock_index_rel where index_code='{}'""".format(index_code)
        tamp_fund_session.execute(sql)

def delete_index_daily_price(index_code, market, rcd_date):
    table_name = 'index_trade_daily_{}'.format(market.lower())
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """DELETE FROM {} WHERE index_code='{}' and trade_date='{}'""".format(table_name, index_code, rcd_date)
        cur = tamp_fund_session.execute(sql)

def get_all_index():
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT * from stock_index"""
        cur = tamp_fund_session.execute(sql)
        return cur.fetchall()

def get_index_daily_price_as_df(index_code, market):
    table_name = get_index_price_table_name(market)
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT `id`, `index_code`, `trade_date`, `open`, `high`, `low`, `close`, `chg`, `vol`, `amount` FROM {} WHERE index_code='{}' order by trade_date asc""".format(table_name, index_code)
        cur = tamp_fund_session.execute(sql)
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['id', 'index_code', 'trade_date', 'open', 'high', 'low', 'close', 'chg', 'vol', 'amount']).dropna(how='any')

        return df

def get_index_price_table_name(market):
    if market == 'CSI':
        return 'index_trade_daily_csi'
    if market == 'SSE':
        return 'index_trade_daily_sse'
    if market == 'SZSE':
        return 'index_trade_daily_szse'




# etf
def get_all_etf():
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT * from basic_etf_info"""
        cur = tamp_fund_session.execute(sql)
        return cur.fetchall()

def delete_etf_daily_price(etf_code, rcd_date):
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """DELETE FROM etf_trade_daily WHERE stock_code='{}' and trade_date='{}'""".format(etf_code, rcd_date)
        cur = tamp_fund_session.execute(sql)

def get_etf_daily_price_as_df(etf_code):
    with TAMP_SQL(stock_engine) as tamp_fund:
        tamp_fund_session = tamp_fund.session
        sql = """SELECT `id`, `stock_code`, `trade_date`, `open`, `high`, `low`, `close`, `chg`, `vol`, `amount` FROM etf_trade_daily WHERE stock_code='{}' order by trade_date asc""".format(etf_code)
        cur = tamp_fund_session.execute(sql)
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['id', 'stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'chg', 'vol', 'amount']).dropna(how='any')

        return df