# -*- coding: utf-8 -*-
# @Time    : 2023/8/28


if __name__ == "__main__":
    '''
    假设你有一个数组prices，长度为n，其中prices[i]是某只股票在第i天的价格，请根据这个价格数组，返回买卖股票能获得的最大收益
    1. 你最多可以对该股票有两笔交易操作，一笔交易代表着一次买入与一次卖出，但是再次购买前必须卖出之前的股票
    2. 如果不能获取收益，请返回0
    3. 假设买入卖出均无手续费
    '''
    n = int(input())
    prices = list(map(int, input().split()))
    
    buy1 = buy2 = -prices[0]
    sell1 = sell2 = 0
    for i in range(1, n):
        # 第一次买入时，赚取的最大利益为支出的费用，即 - price[i]  或者暂不执行第一次买入
        buy1 = max(buy1, -prices[i])
        
        # 第一次卖出时，赚取的最大利益为第一次买入时支出的费用与卖出时股票的价值和，或者暂不执行第一次卖出
        sale1 = max(sale1, buy1 + prices[i])
        
        # 第二次买入时，赚取的最大利益为第一次卖出时获取的最大利益与此时的支出的费用或暂不执行第二次的买入
        buy2 = max(buy2, sale1 - prices[i])
        
        # 第二次卖出时，赚取的最大利益为第二次买入时的最大利益与卖出时股票的价值和或暂不执行第二次卖出
        sale2 = max(sale2, buy2 + prices[i])

    print(sell2)
