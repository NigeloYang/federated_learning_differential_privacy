# -*- coding: utf-8 -*-
# @Time    : 2023/10/23
from typing import List


class Solution:
    def weightGrowth(self, weights: List[int]) -> List[int]:
        if not weights:
            return []
        
        growth = []
        for i in range(len(weights)):
            j = i + 1
            while j < len(weights):
                if weights[j] > weights[i]:
                    growth.append(j-i)
                    break
                j += 1
            if j >= len(weights):
                growth.append(-1)
        return growth
    
if __name__ == "__main__":
    print(Solution().weightGrowth([30,30,30,30,30]))
    print(Solution().weightGrowth([30,40,50,60]))
