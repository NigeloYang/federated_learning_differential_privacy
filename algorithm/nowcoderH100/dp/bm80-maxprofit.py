# -*- coding: utf-8 -*-
# @Time    : 2023/9/26
from typing import List


class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        minp = int(1e9)
        maxp = 0
        for price in prices:
            maxp = max(price-minp,maxp)
            minp = min(price,minp)
        return maxp
    
if __name__ == "__main__":
    print()
