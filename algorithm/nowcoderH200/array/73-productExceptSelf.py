# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
from typing import List


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        ans = [1] * len(nums)
        count = 1
        for i in range(2, len(nums)):
            count *= nums[i - 2]
            ans[i] = count
        count2 = 1
        for i in range(len(nums) - 3, -1, -1):
            count2 *= nums[i + 2]
            ans[i] *= count2
        
        return ans
    
if __name__ == "__main__":
    print(Solution().productExceptSelf([1,2,3,4,5]))
