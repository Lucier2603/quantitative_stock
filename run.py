import tushare

from strategy.special_flag import deviation_degree_on_ma10
from strategy.strategy_deprecated import strategy_big_rise_after_big_fail
from strategy.strategy_stock import filter_strategy, strategy_more_than_ma_30, strategy_rise_high_vol_vs_down_low_vol, \
    strategy_after_big_increase, strategy_find_trend_A, strategy_find_trend_B, strategy_back_to_ma
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
    p_start_date = (max_date[0]+datetime.timedelta(days=1)).strftime('%Y%m%d')
    p_end_date = (datetime.date.today()+datetime.timedelta(days=1)).strftime('%Y%m%d')

    # p_start_date = '20220907'
    # p_end_date = '20220930'

    # 更新股票信息
    # update_month_stock_price_ts(p_start_date, p_end_date, pro)
    # 更新指数信息
    # update_daily_index_price(p_start_date, p_end_date, pro)
    # 更新etf信息
    # update_daily_etf_price(p_start_date, p_end_date, pro)

    # 指定日期范围
    start_strategy_time = datetime.date.today() + datetime.timedelta(days=-2)
    end_strategy_time = datetime.date.today()
    # start_strategy_time = datetime.date(2022, 11, 7)
    # end_strategy_time = datetime.date(2022, 11, 10)



    # 上穿20日线
    filter_strategy(strategy_more_than_ma_30, '上穿20日线', start_strategy_time, end_strategy_time)
    # 寻找趋势 A  (20日趋势线)
    # filter_strategy(strategy_find_trend_A, '寻找趋势A', start_strategy_time, end_strategy_time)
    # 寻找趋势 B  (简单策略：20日都在10日线上)
    filter_strategy(strategy_find_trend_B, '寻找趋势B', start_strategy_time, end_strategy_time)
    # todo 策略 缩量回调到ma20 ma30
    filter_strategy(strategy_back_to_ma, '均线回调', start_strategy_time, end_strategy_time)
    # 大阳后小幅回调
    filter_strategy(strategy_after_big_increase, '大阳回调', start_strategy_time, end_strategy_time)
    # 交易量 阳大阴小
    # filter_strategy(strategy_rise_high_vol_vs_down_low_vol, '交易量阳大阴小', start_strategy_time, end_strategy_time)
    # 小阳堆积
    # filter_strategy(strategy_successive_small_up, '小阳堆积', start_strategy_time, end_strategy_time)


    '''
    仅供研究用
    '''
    # start_strategy_time = datetime.date.today() + datetime.timedelta(days=-2)
    # end_strategy_time = datetime.date.today()
    start_strategy_time = datetime.date(2021, 10, 12)
    end_strategy_time = datetime.date(2022, 10, 13)
    # 10日线偏离度
    # filter_strategy(deviation_degree_on_ma10, 'flag/10日线偏离度', start_strategy_time, end_strategy_time)
    # 先大阴后大阳
    # filter_strategy(strategy_big_rise_after_big_fail, 'test/先大阴后大阳', start_strategy_time, end_strategy_time)
    # 龙头战法 仅供研究用
    # filter_strategy(strategy_first_fall_in_successive_up, '研究策略\\龙头战法', start_strategy_time, end_strategy_time)
    # 缩量大跌 仅供研究用
    # filter_strategy(strategy_fall_with_high_chg_and_low_vol, '研究策略\\缩量大跌', start_strategy_time, end_strategy_time)


