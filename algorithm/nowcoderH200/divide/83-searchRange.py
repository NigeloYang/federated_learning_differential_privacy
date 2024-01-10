# -*- coding: utf-8 -*-
# @Time    : 2023/10/27

'''
农场里有一群牛，每头牛都有一个体重，这些体重按照从大到小的顺序排列在一个 n 的牛棚中。农场主人想要找出特定体重的牛在牛棚中的起始位置和结束位置。

给你一个非递增的整数数组 weights，表示牛棚中牛的体重，和一个整数 target，表示要找的牛的体重。请你找出给定体重的牛在牛棚中的开始位置和结束位置。

如果牛棚中不存在体重为 target 的牛，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
'''
from typing import List


class Solution:
    def searchRange(self, weights: List[int], target: int) -> List[int]:
        if not weights:
            return [-1, -1]
        
        l, r = 0, len(weights) - 1
        
        while l <= r:
            m = (l + r) // 2
            
            if weights[m] < target:
                r = m - 1
            elif weights[m] > target:
                l = m + 1
            else:
                nl, nr = m, m
                while nl >= 0 and weights[nl] == target:
                    nl -= 1
                while nr <= len(weights) - 1 and weights[nr] == target:
                    nr += 1
                
                return [nl + 1, nr - 1]
        return [-1, -1]
    
if __name__ == "__main__":
    print(Solution().searchRange([500,500,400,300,300,300,200,200,100],300))
