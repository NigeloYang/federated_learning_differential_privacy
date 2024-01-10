# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
from typing import List


class Solution:
    def findPeakElement(self, heights: List[int]) -> int:
        if not heights:
            return -1
        
        maxi = 0
        for i in range(len(heights)):
            if heights[i] > heights[maxi]:
                maxi = i
        return maxi


if __name__ == "__main__":
    print(Solution().findPeakElement([1, 2, 1, 3, 5, 6, 4]))
