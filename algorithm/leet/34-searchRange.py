# -*- coding: utf-8 -*-
# @Time    : 2023/12/8
from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        
        i, j = -1, -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                i = m
                r = m - 1
            elif nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                j = m
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        
        return [i, j]
    

    
    def research(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (right + left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    def searchRange2(self, nums: List[int], target: int) -> List[int]:
        start = self.research(nums, target)
        end = self.research(nums, target + 1) - 1
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        return [start, end]


if __name__ == "__main__":
    print(Solution().searchRange([5, 7, 7, 8, 8, 10], 8))
