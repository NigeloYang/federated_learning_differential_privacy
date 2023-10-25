# -*- coding: utf-8 -*-
# @Time    : 2023/9/26
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(len(prices) - 1):
            temp = prices[i + 1] - prices[i]
            if temp > 0:
                profit += temp
        
        return profit
    
    def maxProfit2(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        
        profit = [0] * len(prices)
        buy = [0] * len(prices)
        buy[0] = -prices[0]
        for i in range(1, len(prices)):
            buy[i] = max(buy[i - 1], profit[i - 1] - prices[i])
            profit[i] = max(profit[i-1], prices[i] + buy[i - 1])

        return profit[-1]

if __name__ == "__main__":
    print(Solution().maxProfit2([7,1,5,3,6,4]))
