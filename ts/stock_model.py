# coding=utf-8

from sqlalchemy import Column, DECIMAL, Float, Date, DateTime, String, TIMESTAMP, \
    Text, text, Integer, FetchedValue, BIGINT


from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
Base2 = declarative_base()

class BasicStockInfo(Base):
    __tablename__ = 'basic_stock_info'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    stock_name = Column(String(32))
    stock_symbol = Column(String(32))
    area = Column(String(32))
    industry = Column(String(32))
    market = Column(String(16))
    is_hk = Column(String(32))

class StockTradeDaily00x(Base):
    __tablename__ = 'stock_trade_daily_00x'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class StockTradeDaily30x(Base):
    __tablename__ = 'stock_trade_daily_30x'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class StockTradeDaily60x(Base):
    __tablename__ = 'stock_trade_daily_60x'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class StockTradeDaily83x(Base):
    __tablename__ = 'stock_trade_daily_83x'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class StockTradeDaily688(Base):
    __tablename__ = 'stock_trade_daily_688'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)


class StockIndex(Base):
    __tablename__ = 'stock_index'

    id = Column(BIGINT(), primary_key=True, index=True)
    index_code = Column(String(16))
    index_name = Column(String(32))
    market = Column(String(16))
    category = Column(String(32))
    stock_weight = Column(Text)
    trade_date = Column(Date)


class StockIndexRel(Base):
    __tablename__ = 'stock_index_rel'

    id = Column(BIGINT(), primary_key=True, index=True)
    index_code = Column(String(16))
    stock_code = Column(String(32))
    weight = Column(Float)


class IndexTradeDailyCSI(Base):
    __tablename__ = 'index_trade_daily_csi'

    id = Column(BIGINT(), primary_key=True, index=True)
    index_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class IndexTradeDailySSE(Base):
    __tablename__ = 'index_trade_daily_sse'

    id = Column(BIGINT(), primary_key=True, index=True)
    index_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)

class IndexTradeDailySZSE(Base):
    __tablename__ = 'index_trade_daily_szse'

    id = Column(BIGINT(), primary_key=True, index=True)
    index_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)




'''
ETF
'''
class BasicETFInfo(Base):
    __tablename__ = 'basic_etf_info'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    stock_name = Column(String(32))
    fund_type = Column(String(32))


class ETFTradeDaily(Base):
    __tablename__ = 'etf_trade_daily'

    id = Column(BIGINT(), primary_key=True, index=True)
    stock_code = Column(String(16))
    trade_date = Column(Date)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    chg = Column(Float)