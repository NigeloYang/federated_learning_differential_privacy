# -*- coding: utf-8 -*-
# @Time    : 2023/9/26
from typing import List


class Solution:
    def maxProfit2(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
    
        buy1 = buy2 = -prices[0]
        sell1 = sell2 = 0
        for i in range(1, len(prices)):
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, prices[i] + buy1)
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, prices[i] + buy2)
        return sell2

if __name__ == "__main__":
    print(Solution().maxProfit2([7, 1, 5, 3, 6, 4]))
