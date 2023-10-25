# -*- coding: utf-8 -*-
# @Time    : 2023/9/30
from typing import List

'''接雨水问题'''


class Solution:
    def maxWater(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0
        res = left = maxleft = maxright = 0
        right = len(height) - 1
        while left <= right:
            maxleft = max(maxleft,height[left])
            maxright = max(maxright,height[right])
            if maxleft < maxright:
                res += maxleft - height[left]
                left += 1
            else:
                res += maxright - height[right]
                right -= 1
        return res


if __name__ == "__main__":
    print(Solution().maxWater([3, 1, 2, 5, 2, 4]))
