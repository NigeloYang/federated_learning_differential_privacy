# -*- coding: utf-8 -*-
# @Time    : 2023/9/28
from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        if not height:
            return 0
        l, r, maxa = 0, len(height), 0
        while l < r:
            heig = min(height[l], height[r - 1])
            maxa = max(maxa, (r-1-l) * heig)
            if height[l] <= height[r - 1]:
                l += 1
            else:
                r -= 1
        return maxa


if __name__ == "__main__":
    print(Solution().maxArea([1,7,3,2,4,5,8,2,7]))
