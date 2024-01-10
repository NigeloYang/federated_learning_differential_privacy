# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        
        l, r = 0, len(nums) - 1
        
        while l < r:
            m = (r + l) // 2
            if nums[m] == target:
                return m
            elif nums[m] > target and nums[r] <= target:
                l = m + 1
            else:
                r = m - 1
        return -1
    
if __name__ == "__main__":
    print(Solution().search([4,3,2,1,7,6,5],1))
