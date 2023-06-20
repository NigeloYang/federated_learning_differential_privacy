# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

class Solution:
    def maxProfit(self, prices) -> int:
        buy, profit = float('+inf'), 0
        for price in prices:
            buy = min(buy, price)
            profit = max(profit, price - buy)
        return profit


if __name__ == "__main__":
    print(Solution().maxProfit([7,1,5,3,6,4]))
