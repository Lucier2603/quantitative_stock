import tushare

from ts.stock_crawler import update_month_stock_price_ts, update_daily_index_price, update_daily_etf_price
from ts.stock_data_service import get_max_stock_rcd_date
import datetime

if __name__ == '__main__':
    tushare.set_token('702d425922f79bd2ebda78a369f6fe1a40982bbf6a51e00c61bec879')
    pro = tushare.pro_api()

    # sync_all_stock()
    # sync_all_index()
    # sync_all_ETF()

    max_date = get_max_stock_rcd_date()
    p_start_date = max_date[0].strftime('%Y%m%d')
    p_end_date = datetime.date.today().strftime('%Y%m%d')

    # p_start_date = '20220907'
    # p_end_date = '20220930'

    # 更新股票信息
    update_month_stock_price_ts(p_start_date, p_end_date, pro)
    # 更新指数信息
    update_daily_index_price(p_start_date, p_end_date, pro)
    # 更新etf信息
    update_daily_etf_price(p_start_date, p_end_date, pro)