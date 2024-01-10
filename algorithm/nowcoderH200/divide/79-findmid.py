# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
from typing import List


class Solution:
    def findMin(self, heights: List[int]) -> int:
        if not heights:
            return 0
        l, r = 0, len(heights) - 1
        
        if heights[l] > heights[r]:
            return heights[r]
        
        while l < r:
            if r-l == 1:
                break
            m = (l + r) // 2
            if heights[l] >= heights[m]:
                l = m
            else:
                r = m
        return heights[l]


if __name__ == "__main__":
    print(Solution().findMin([5, 4, 3, 2, 1, 7, 6]))
    print(Solution().findMin([0, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]))
    print((0+1)//2)
