# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List


class Solution:
    '''常规先排序，在找'''
    
    def findKthSmallest(self, weights: List[int], k: int) -> int:
        if not weights:
            return 0
        weights.sort()
        return weights[k - 1]
    
    def findKthSmallest2(self, weights: List[int], k: int) -> int:
        if not weights:
            return 0
        
        count = [0] * 5000
        for weight in weights:
            count[weight] += 1
        
        for i in range(5000):
            if count[i] > 0:
                count[i] -= 1
                k -= 1
                if k == 0:
                    return i


if __name__ == "__main__":
    print(Solution().findKthSmallest2([600,500,800,700,550,650],3))
