
import pandas as pd

def diff():
    big = pd.read_csv('D:\work\寻找趋势B_bs_df_total_1.csv')
    small = pd.read_csv('D:\work\寻找趋势B_bs_df_total_2.csv')

    sbig = set()
    ssmall = set()

    for i in big.index:
        r = big.loc[i]
        key = r['stock_code'] + ' ' + r['stock_name']
        sbig.add(key)

    for i in small.index:
        r = small.loc[i]
        key = r['stock_code'] + ' ' + r['stock_name']
        ssmall.add(key)

    s3 = sbig.intersection(ssmall)
    s4 = sbig-s3

    for s in s4:
        print(s)


if __name__ == '__main__':
    diff()